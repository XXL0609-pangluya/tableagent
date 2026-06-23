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
# Tier 2b — independent COMPUTE check (verifier writes & runs its own pandas)
# --------------------------------------------------------------------------

_COMPUTE_SYSTEM = """You are an INDEPENDENT verifier for table question answering,
running on a different model than the solver. Solve the question YOURSELF from
scratch — do NOT trust any prior answer. Write pandas code against the DataFrame
`df` (every column is a string; cast with care, e.g. strip commas/units before
int/float). Read the question precisely:
- distinguish what is being counted (rows vs distinct values vs an entity buried
  inside a cell like "Company (Country)"),
- match the answer TYPE the question asks for (a name/label vs a number; one value
  vs a list),
- exclude Total/summary rows and handle blank or "–"/"—" cells when aggregating.

Your code MUST assign the final answer to a variable `answer` (a scalar, or a list
for multi-item answers). Return ONLY one python code block, nothing else."""


def _df_view(df: pd.DataFrame, max_rows: int = 8, max_chars: int = 2000) -> str:
    cols = ", ".join(f"{c!r}" for c in df.columns)
    lines = [f"Columns ({len(df.columns)}): {cols}", f"Row count: {len(df)}", "", "Sample rows:"]
    for i in range(min(max_rows, len(df))):
        cells = " | ".join(f"{c}={str(df.iloc[i][c])}" for c in df.columns)
        lines.append(f"  [{i}] {cells}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n…(truncated)"


def _extract_code(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # no fence — assume the whole thing is code if it looks like it
    return text.strip() if ("answer" in text and "=" in text) else ""


def compute_check(
    client: LLMClient,
    question: str,
    items: list[str],
    df: pd.DataFrame,
    *,
    max_tokens: int = 1500,
    timeout_s: float = 10.0,
) -> VerifyResult:
    """Verifier independently recomputes the answer and compares to the solver's.

    - match  -> ok=True  (corroborated by a second model + second computation)
    - differ -> ok=False (flag, carry the recomputed value as an advisory hint)
    - verifier failed/abstained -> ok=True, source 'none' (no signal)
    """
    user = (
        f"Question: {question}\n\n"
        f"{_df_view(df)}\n\n"
        "Write pandas code that sets `answer`."
    )
    try:
        resp = client.chat(
            messages=[{"role": "system", "content": _COMPUTE_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — never block the run
        return VerifyResult(ok=True, source="none", model=client.config.model, raw=f"compute skipped: {exc}")

    code = _extract_code(resp.text or "")
    if not code:
        return VerifyResult(ok=True, source="none", model=client.config.model, raw="no code emitted")

    exec_res = run_code(code, df.copy(), timeout_s=timeout_s)
    if not exec_res.ok or exec_res.answer is None:
        return VerifyResult(ok=True, source="none", model=client.config.model,
                            raw=f"compute error: {exec_res.error or 'answer not set'}")

    from .tools.wtq_tools import coerce_items  # local import to avoid cycle
    v_items = coerce_items(exec_res.answer)
    if not v_items:
        return VerifyResult(ok=True, source="none", model=client.config.model, raw="empty recompute")

    if answers_match(items, v_items):
        return VerifyResult(ok=True, source="compute", model=client.config.model,
                            recomputed=v_items, compute_match=True,
                            verifier_code=code, verifier_stdout=exec_res.stdout[:400])

    issue = (
        f"An independent recomputation (different model) produced {v_items}, "
        f"which differs from your answer {items}."
    )
    return VerifyResult(ok=False, issues=[issue], source="compute", model=client.config.model,
                        recomputed=v_items, compute_match=False,
                        verifier_code=code, verifier_stdout=exec_res.stdout[:400],
                        fix_hint="Re-read the question and re-derive with run_python; "
                        "decide which interpretation matches the question.")


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
    use_llm: bool = True,
) -> VerifyResult:
    """Run ALL applicable checks and aggregate. Crucially, the A/B/C dimensional
    review ALWAYS runs even when the independent recompute agrees — two models
    can share the same blind spot (e.g. both drop a unit, both count rows instead
    of distinct values), so answer-agreement alone is not sufficient.

      1. deterministic rules (no model) — short-circuits if it fires (high precision)
      2. A/B/C dimensional review (reads table) — ALWAYS runs
      3. independent COMPUTE check (verifier runs its own pandas) — if df given
    A concern from ANY tier flags the answer. The recompute's value is also carried
    as an extra vote for finalize, whether or not it agreed.
    """
    if not items:
        return VerifyResult(ok=True, source="none")

    det = deterministic_issues(question, items)
    if df is not None:
        det += cell_substring_issues(items, df)
    if det:
        return VerifyResult(ok=False, issues=det, source="deterministic", axis="A")

    if client is None or not use_llm:
        return VerifyResult(ok=True, source="none")

    # Tier 2: A/B/C dimensional review (always on)
    review = llm_check(client, question, items, table_view, evidence_summary)

    # Tier 3: independent recompute (extra evidence)
    compute = (
        compute_check(client, question, items, df)
        if df is not None else VerifyResult(ok=True, source="none")
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
    if compute.source == "compute" and not compute.ok:
        ok = False
        issues += compute.issues
        sources.append("compute")

    return VerifyResult(
        ok=ok,
        issues=issues,
        fix_hint=compute.fix_hint or review.fix_hint,
        source="+".join(sources) if sources else ("compute" if compute.recomputed else "none"),
        axis=axis,
        model=client.config.model,
        recomputed=compute.recomputed,
        compute_match=compute.compute_match,
    )


_AXIS_HINT = {
    "A": "Form/precision: copy the value EXACTLY as the cell shows it (no rounding, "
         "no added/removed units, keep multi-part cells whole, don't expand abbreviations).",
    "B": "Intent: make sure the answer TYPE matches the question (name vs number, single vs list).",
    "C": "Counting: de-duplicate, skip blank/'–' cells, and handle any Total/summary row correctly.",
}


def build_debate_prompt(vr: VerifyResult, solver_answer: list[str]) -> str:
    """Structured debate-mode prompt injected after a verification flag.

    Shows the verifier's code evidence (if available) so the generator can
    evaluate it critically instead of just capitulating on a text assertion.
    The generator MUST run its own run_python before resubmitting — that code
    is the evidence that decides who is right.
    """
    lines = [
        "═══ VERIFIER REPORT ═══",
        f"An independent verifier (different model: {vr.model}) reviewed your answer "
        f"{solver_answer} and raised concerns.",
        "",
    ]

    # Show the verifier's actual code so the generator can judge it
    if vr.verifier_code:
        lines += [
            f"The verifier ran this pandas code and got {vr.recomputed}:",
            "```python",
            vr.verifier_code.strip(),
            "```",
        ]
        if vr.verifier_stdout:
            lines += [f"stdout: {vr.verifier_stdout.strip()}"]
        lines.append("")

    # A/B/C axis finding
    if vr.axis and vr.axis in _AXIS_HINT:
        lines.append(f"Axis {vr.axis} concern — {_AXIS_HINT[vr.axis]}")
    if vr.issues:
        lines.append("Specific concern: " + "; ".join(vr.issues))
    if vr.fix_hint:
        lines.append(f"Verifier suggests re-checking: {vr.fix_hint}")

    lines += [
        "",
        "═══ YOUR TURN ═══",
        "You have been granted extra steps for this discussion.",
        "You MUST run run_python to verify before resubmitting. Two paths:",
        "  • Defend your answer: run code that proves YOUR derivation is correct, then resubmit the SAME answer.",
        "  • Accept the correction: run code that confirms the verifier's logic is right, then resubmit the corrected answer.",
        "Do NOT change your answer without running code. If you run code and both answers look plausible,",
        "trust the one whose derivation is more directly supported by the table.",
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
