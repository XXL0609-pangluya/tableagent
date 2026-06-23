"""Phase 2c answer verifier — re-reads the question and checks the proposed
answer against the table along three dimensions, using an INDEPENDENT model
(a different LLM than the solver, to avoid same-model blind spots).

Two precision tiers, checked in order:
  1. deterministic_issues(): rule-based, no LLM (anchor inclusion, cardinality).
  2. llm_check(): table-aware checker on a separate model. It re-reads the
     question and inspects the table view to judge three axes:
       A. FORM/PRECISION  — verbatim cell value, rounding, units, multi-part cell
       B. INTENT          — label vs value, single vs list
       C. COUNTING/AGG    — dedup, blank/"–" cells, Total/summary rows

The verifier flags problems and says WHAT to re-check; the agent re-derives.
A candidate pool in the agent loop guarantees a noisy verifier can never replace
a good answer with a worse one (it can only trigger a re-derivation).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .evaluator import check_denotation, to_value_list
from .llm import LLMClient
from .sandbox import run_code


@dataclass
class VerifyResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    fix_hint: str = ""
    source: str = "none"   # "deterministic" | "compute" | "llm" | "llm+compute" | "none"
    axis: str = ""          # "A" | "B" | "C" | ""
    model: str = ""
    recomputed: list[str] = field(default_factory=list)  # verifier's own answer (compute tier)
    compute_match: Optional[bool] = None  # True=agreed, False=disagreed, None=abstained/no df
    # Code evidence the verifier actually executed — surfaced to the generator in debate mode
    # so the generator can read it, judge it, and run counter-code instead of just capitulating.
    verifier_code: str = ""
    verifier_stdout: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": self.issues, "fix_hint": self.fix_hint,
                "source": self.source, "axis": self.axis, "model": self.model,
                "recomputed": self.recomputed, "compute_match": self.compute_match}


def answers_match(a: list[str], b: list[str]) -> bool:
    """Order-insensitive denotation match (reuses the official evaluator logic)."""
    try:
        return check_denotation(to_value_list(a), to_value_list(b))
    except Exception:  # noqa: BLE001
        return _norm_cmp(" ".join(a)) == _norm_cmp(" ".join(b))


# --------------------------------------------------------------------------
# Tier 1 — deterministic, rule-based checks (no LLM, high precision)
# --------------------------------------------------------------------------

def _question_anchor(question: str) -> Optional[str]:
    """Extract a reference entity the answer should usually EXCLUDE
    ("other than X", "same as the first entry (X)", "besides X")."""
    q = question.strip()
    m = re.search(r"\(([^)]{2,})\)", q)
    if m:
        return m.group(1).strip().lower()
    m = re.search(r"\b(?:other than|besides|except|excluding|apart from)\s+(.+?)[\?\.\,]?$", q, re.I)
    if m:
        return m.group(1).strip().lower()
    return None


def _norm_cmp(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("'", "").replace('"', "").strip().lower())


def _expects_single(question: str) -> Optional[bool]:
    q = question.strip().lower()
    if re.search(r"\b(how many|number of)\b", q):
        return True
    if re.search(r"\b(list|name all|which ones|what are all)\b", q):
        return False
    if re.match(r"^(which|who|what|where|when)\b", q) and " and " not in q:
        return True
    return None


def _is_number(s: str) -> bool:
    try:
        float(s.replace(",", "").strip())
        return True
    except ValueError:
        return False


def cell_substring_issues(items: list[str], df: "pd.DataFrame") -> list[str]:
    """A-axis: flag a single answer that is only PART of a cell (never a full cell).

    WTQ answers are usually the complete cell value, so a word-bounded sub-phrase
    that never appears as a whole cell (e.g. '4x400 m' when the cell is
    '4x400 m relay') is likely a truncation. Conservative: skip numbers, and skip
    if the answer DOES appear as a complete cell somewhere (then it's legitimate).
    """
    if len(items) != 1:
        return []
    ans = items[0].strip()
    if not ans or _is_number(ans):
        return []
    al = ans.lower()
    pat = re.compile(r"(^|\s)" + re.escape(al) + r"(\s|$)")
    extensions: set[str] = set()
    for col in df.columns:
        for val in df[col].astype(str).unique():
            v = val.strip()
            vl = v.lower()
            if vl == al:
                return []  # answer is a complete cell somewhere -> legitimate
            if len(vl) > len(al) and pat.search(vl):
                extensions.add(v)
    if not extensions:
        return []
    sample = sorted(extensions)[:3]
    return [
        f"Your answer '{ans}' never appears as a complete cell — it is only part of "
        f"cell(s) like {sample}. WTQ answers are usually the FULL cell text; confirm "
        f"whether the answer should be the complete value."
    ]


def deterministic_issues(question: str, items: list[str]) -> list[str]:
    issues: list[str] = []
    if not items:
        return issues
    anchor = _question_anchor(question)
    if anchor and len(items) > 1:
        anchor_key = _norm_cmp(anchor)
        for it in items:
            it_key = _norm_cmp(it)
            # exact match, or the item is the leading token(s) of the captured phrase
            # (e.g. anchor "france won" still matches item "france")
            if it_key and (it_key == anchor_key or anchor_key.startswith(it_key + " ")):
                issues.append(
                    f"The answer includes the reference entity '{it}' named in the question; "
                    "questions like 'other than / same as (X)' should EXCLUDE it and return only the OTHERS."
                )
                break
    if _expects_single(question) is True and len(items) >= 3:
        issues.append(
            f"The question seems to expect a single answer but {len(items)} items were returned: {items}. "
            "Re-check whether the filter is too broad."
        )
    return issues


# --------------------------------------------------------------------------
# Tier 2 — table-aware LLM check on an independent model
# --------------------------------------------------------------------------

_LLM_SYSTEM = """You are an INDEPENDENT verifier for table question answering. You
are a different model from the one that produced the answer. Re-read the question
carefully, look at the table, and judge whether the proposed answer is correct
along THREE axes:

A. FORM / PRECISION — Is each answer item copied EXACTLY as it appears in the
   relevant table cell? Flag if the answer rounds a number the table didn't round,
   adds/removes a unit, expands an abbreviation (e.g. "United States" when the cell
   says "USA"), or drops part of a multi-part cell (e.g. a name that also carries
   dates/qualifiers).
B. INTENT — Does the answer TYPE match what the question asks? A question wanting a
   name/label must not be answered with a number (and vice versa); a single-answer
   question must not return a list.
C. COUNTING / AGGREGATION — For counts/sums/averages/extremes, were duplicates
   de-duplicated when needed, blank or "–"/"—" cells handled, and any Total/summary
   row excluded (or included) as the question requires?

Rules:
- Use the table to judge; do NOT invent rows or values you cannot see.
- If the answer is fine on all three axes, pass.
- If not, name the single most important axis (A/B/C) and say briefly what to
  re-check. Do NOT just assert a replacement number — describe the check.

Respond with JSON only:
{"pass": true|false, "axis": "A"|"B"|"C"|null, "issue": "short", "fix_hint": "what to re-check"}
"""


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"pass": True, "axis": None, "issue": "", "fix_hint": ""}


def llm_check(
    client: LLMClient,
    question: str,
    items: list[str],
    table_view: str = "",
    evidence_summary: str = "",
    *,
    max_tokens: int = 1024,  # headroom for reasoning models (content comes after CoT)
) -> VerifyResult:
    user = (
        f"Question: {question}\n\n"
        f"Proposed answer items: {json.dumps(items, ensure_ascii=False)}\n\n"
        f"Solver's evidence: {evidence_summary or '(none)'}\n\n"
        f"Table:\n{table_view or '(table not provided)'}\n"
    )
    try:
        resp = client.chat(
            messages=[{"role": "system", "content": _LLM_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — verifier must never block the run
        return VerifyResult(ok=True, source="none", model=client.config.model, raw=f"skipped: {exc}")

    raw = resp.text or ""
    parsed = _parse_json(raw)
    ok = bool(parsed.get("pass", True))
    issue = str(parsed.get("issue") or "").strip()
    axis = str(parsed.get("axis") or "").strip().upper()
    axis = axis if axis in ("A", "B", "C") else ""
    fix_hint = str(parsed.get("fix_hint") or "").strip()
    if ok or not issue:
        return VerifyResult(ok=True, source="none", model=client.config.model, raw=raw[:400])
    return VerifyResult(ok=False, issues=[issue], fix_hint=fix_hint, source="llm",
                        axis=axis, model=client.config.model, raw=raw[:400])


# --------------------------------------------------------------------------
# Tier 2b — AUDIT check: reviews the generator's code history for specific flaws
# (does NOT re-answer from scratch — only audits the generator's own reasoning)
# --------------------------------------------------------------------------

_AUDIT_SYSTEM = """You are a meticulous code auditor reviewing a student's work on a table question-answering task.

IMPORTANT: You are NOT re-answering the question. You are AUDITING the student's reasoning process.

You will receive:
1. The question
2. A sample of the table
3. The student's proposed answer
4. The student's run_python steps (code + outputs)

Your job: find ONE specific, concrete flaw in the student's code or reasoning, if any. 

VALID flaws to flag (all pointing to a specific step):
- Wrong aggregation: "Step 2 used .count() but the question needs .nunique() for distinct values"
- Missing exclusion: "Step 3 didn't filter out the Total/summary row (e.g. row where Name='Total')"
- Wrong column: "Step 1 filtered by column 'Winner' but the question asks about 'Runner-up'"
- Off-by-one: "Step 2's date filter is >= instead of >, including the boundary date itself"
- Truncated cell: "The answer '4x400 m' is a substring of cell '4x400 m relay' — use the full cell"
- Type mismatch: "The answer is a number but the question asks for a name/label"

INVALID — do NOT do these:
- Do NOT compute the answer yourself from scratch
- Do NOT say "the answer should be X" without pointing to a specific flawed step
- Do NOT invent flaws; if the student's work looks sound, say it passes
- Do NOT flag stylistic issues — only flag logical/factual errors

Respond with JSON only:
{
  "flawed": true|false,
  "step": "which step is flawed, e.g. 'run_python step 2' or 'final answer format'",
  "flaw": "one sentence describing the concrete error",
  "test": "one line of pandas code the student can run to verify this claim (or empty string)"
}"""


def audit_check(
    client: LLMClient,
    question: str,
    items: list[str],
    code_history: str,
    table_view: str = "",
    *,
    max_tokens: int = 1024,
) -> VerifyResult:
    """Audit the generator's own code history for a specific logical flaw.

    Returns the flaw description + a one-liner test the generator can run to
    verify the claim. Does NOT compute a new answer.
    """
    user = (
        f"Question: {question}\n\n"
        f"Student's proposed answer: {items}\n\n"
        f"Table sample:\n{table_view or '(not provided)'}\n\n"
        f"Student's reasoning steps:\n{code_history or '(no code steps recorded)'}\n"
    )
    try:
        resp = client.chat(
            messages=[{"role": "system", "content": _AUDIT_SYSTEM},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(ok=True, source="none", model=client.config.model,
                            raw=f"audit skipped: {exc}")

    raw = resp.text or ""
    parsed = _parse_json(raw)
    flawed = bool(parsed.get("flawed", False))
    if not flawed:
        return VerifyResult(ok=True, source="audit", model=client.config.model, raw=raw[:300])

    step_desc = str(parsed.get("step") or "").strip()
    flaw_desc = str(parsed.get("flaw") or "").strip()
    test_code = str(parsed.get("test") or "").strip()
    if not flaw_desc:
        return VerifyResult(ok=True, source="none", model=client.config.model, raw=raw[:300])

    issue = f"[{step_desc}] {flaw_desc}" if step_desc else flaw_desc
    return VerifyResult(
        ok=False, issues=[issue], source="audit",
        model=client.config.model,
        verifier_code=test_code,   # reuse field: the test-code to reproduce the claim
        fix_hint=f"Run: {test_code}" if test_code else "",
        raw=raw[:300],
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def verify(
    client: Optional[LLMClient],
    question: str,
    items: list[str],
    *,
    df: Optional[pd.DataFrame] = None,
    table_view: str = "",
    evidence_summary: str = "",
    code_history: str = "",
    use_llm: bool = True,
) -> VerifyResult:
    """Three tiers, all running independently:
      1. deterministic rules (no model) — short-circuits on high-precision hits
      2. A/B/C dimensional review — checks answer quality (form, intent, counting)
      3. AUDIT of the generator's code history — finds specific logical flaws
         (does NOT re-answer; points to a concrete step and provides a test snippet)
    """
    if not items:
        return VerifyResult(ok=True, source="none")

    # Tier 1: deterministic (no model cost)
    det = deterministic_issues(question, items)
    if df is not None:
        det += cell_substring_issues(items, df)
    if det:
        return VerifyResult(ok=False, issues=det, source="deterministic", axis="A")

    if client is None or not use_llm:
        return VerifyResult(ok=True, source="none")

    # Tier 2: A/B/C dimensional review (always on)
    review = llm_check(client, question, items, table_view, evidence_summary)

    # Tier 3: audit the generator's code history for specific logical flaws
    audit = (
        audit_check(client, question, items, code_history, table_view)
        if code_history else VerifyResult(ok=True, source="none")
    )

    issues: list[str] = []
    sources: list[str] = []
    axis = ""
    ok = True
    if not review.ok:
        ok = False
        issues += review.issues
        axis = review.axis or axis
        sources.append("llm")
    if not audit.ok:
        ok = False
        issues += audit.issues
        sources.append("audit")

    final_source = "+".join(sources) if sources else "none"
    return VerifyResult(
        ok=ok,
        issues=issues,
        fix_hint=audit.fix_hint or review.fix_hint,
        source=final_source,
        axis=axis,
        model=client.config.model,
        verifier_code=audit.verifier_code,   # the test snippet from the audit
    )


_AXIS_HINT = {
    "A": "Form/precision: copy the value EXACTLY as the cell shows it (no rounding, "
         "no added/removed units, keep multi-part cells whole, don't expand abbreviations).",
    "B": "Intent: make sure the answer TYPE matches the question (name vs number, single vs list).",
    "C": "Counting: de-duplicate, skip blank/'–' cells, and handle any Total/summary row correctly.",
}


def build_debate_prompt(vr: VerifyResult, solver_answer: list[str]) -> str:
    """Debate-mode prompt: auditor presents a specific claim; generator evaluates it
    critically with code, then decides. The generator's default stance is confidence
    in its own derivation — only change if the specific claim is verified."""
    lines = [
        "═══ AUDITOR REPORT ═══",
        f"An independent auditor (model: {vr.model}) reviewed your reasoning steps "
        f"and raised a specific concern about your answer {solver_answer}.",
        "",
    ]

    # Show the specific audit finding
    if vr.issues:
        lines.append("Specific claim: " + "; ".join(vr.issues))
    if vr.axis and vr.axis in _AXIS_HINT:
        lines.append(f"Category: Axis {vr.axis} — {_AXIS_HINT[vr.axis]}")

    # Show the test code if available
    if vr.verifier_code:
        lines += [
            "",
            "The auditor suggests running this targeted test to verify the claim:",
            "```python",
            vr.verifier_code.strip(),
            "```",
        ]

    lines += [
        "",
        "═══ YOUR TURN (you have extra steps) ═══",
        "You are the expert who worked directly with this table.",
        "Default stance: your answer is PRESUMED CORRECT.",
        "",
        "Run the test above (or equivalent code) and then decide:",
        "  • If the test CONFIRMS the auditor's claim → fix the specific issue and resubmit.",
        "  • If the test REFUTES the claim → resubmit your ORIGINAL answer with",
        "    the test output as evidence. A single assertion without proof is not enough",
        "    to change your answer.",
        "",
        "Do not change your answer based on the auditor's words alone — only on code evidence.",
    ]
    return "\n".join(lines)


def build_verify_feedback(vr: VerifyResult) -> str:
    """Backwards-compatible alias — used for non-compute flags (deterministic / llm-only)."""
    parts = [
        "An independent verifier flagged a possible problem with your submitted answer. "
        "Re-read the question, re-check the table with run_python/search_cells, then submit again. "
        "If after re-checking you remain confident, resubmit the SAME answer.",
    ]
    if vr.axis and vr.axis in _AXIS_HINT:
        parts.append(f"Axis {vr.axis} — {_AXIS_HINT[vr.axis]}")
    if vr.issues:
        parts.append("Concern: " + "; ".join(vr.issues))
    if vr.fix_hint:
        parts.append(f"What to re-check: {vr.fix_hint}")
    return "\n\n".join(parts)
