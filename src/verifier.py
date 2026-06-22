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

from .llm import LLMClient


@dataclass
class VerifyResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    fix_hint: str = ""
    source: str = "none"   # "deterministic" | "llm" | "none"
    axis: str = ""          # "A" | "B" | "C" | ""
    model: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": self.issues, "fix_hint": self.fix_hint,
                "source": self.source, "axis": self.axis, "model": self.model}


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
# Orchestration
# --------------------------------------------------------------------------

def verify(
    client: Optional[LLMClient],
    question: str,
    items: list[str],
    *,
    table_view: str = "",
    evidence_summary: str = "",
    use_llm: bool = True,
) -> VerifyResult:
    """Deterministic checks first (high precision); then independent-model check."""
    if not items:
        return VerifyResult(ok=True, source="none")
    det = deterministic_issues(question, items)
    if det:
        return VerifyResult(ok=False, issues=det, source="deterministic")
    if use_llm and client is not None:
        return llm_check(client, question, items, table_view, evidence_summary)
    return VerifyResult(ok=True, source="none")


_AXIS_HINT = {
    "A": "Form/precision: copy the value EXACTLY as the cell shows it (no rounding, "
         "no added/removed units, keep multi-part cells whole, don't expand abbreviations).",
    "B": "Intent: make sure the answer TYPE matches the question (name vs number, single vs list).",
    "C": "Counting: de-duplicate, skip blank/'–' cells, and handle any Total/summary row correctly.",
}


def build_verify_feedback(vr: VerifyResult) -> str:
    """Message injected into the agent loop when verification flags an issue."""
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
