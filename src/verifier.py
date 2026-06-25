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
    # Audit evidence surfaced to the generator in debate mode.
    # verifier_code    — one-liner pandas snippet the generator should run to test the claim
    # verifier_stdout  — output from any code the verifier ran itself (currently unused)
    # verifier_reasoning — full natural-language explanation of WHY the auditor suspects a flaw
    #   (table observations, logical chain); shown to the generator alongside the test code
    verifier_code: str = ""
    verifier_stdout: str = ""
    verifier_reasoning: str = ""
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
    # Explicit-plural / list questions → expect a LIST, not a single item.
    if "(s)" in q or "for each" in q or \
       re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten|both)\b", q):
        return False
    # "which buildings/nations/players/routes ..." (plural head noun) → a list.
    m = re.match(r"^(which|what)\s+([a-z]+)", q)
    if m:
        noun = m.group(2)
        if noun.endswith("s") and noun not in ("is", "was", "series", "does", "has"):
            return False
    if re.match(r"^(which|who|what|where|when)\b", q) and " and " not in q:
        return True
    return None


def _expects_exactly_one(question: str) -> bool:
    """Strict singular phrasing: the answer must be a SINGLE item.
    High precision — used to flag multi-item answers to clearly-singular questions."""
    q = question.strip().lower()
    if re.search(r"\b(list|name all|which ones|what are all|all of|which.*\bare\b)\b", q):
        return False
    # Explicit plural markers → NOT exactly one: "which season(s)", "what TWO teams",
    # "both", "three monarchs". (Measured: these were the cardinality check's only FPs.)
    if "(s)" in q or re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten|both)\b", q):
        return False
    # explicit singular markers
    if re.search(r"\bthe (only|first|last|single|sole)\b", q):
        return True
    # "name another X", "a different X", "one other X" → exactly one more
    if re.search(r"\b(another|a different|one other)\b", q):
        return True
    if re.search(r"\b(who|whom|whose)\b", q):
        return True
    # "which/what <singular-noun>" anywhere — singular if the noun isn't plural
    m = re.search(r"\b(which|what)\s+([a-z]+)\b", q)
    if m:
        noun = m.group(2)
        if not noun.endswith("s") and noun not in ("is", "was", "are", "were", "many", "kind", "type"):
            return True
    return False


# Nouns after which/what that ask for a NUMBER/temporal value, not an entity label.
_NUMERIC_NOUNS = (
    "year", "number", "percentage", "percent", "amount", "count", "total",
    "time", "score", "rank", "place", "position", "age", "size", "distance",
    "difference", "average", "sum", "many", "much",
)

# Superlative / argmax markers: "which X had the MOST points" wants the entity X,
# almost never the measure (points) or a numeric id.
_SUPERLATIVE_RE = re.compile(
    r"\b(most|least|fewest|highest|lowest|largest|smallest|greatest|biggest|"
    r"longest|shortest|best|worst|top|maximum|minimum|earliest|latest)\b"
)

# Verbs/auxiliaries that can follow "which/what" without being the entity noun.
_COPULAS = (
    "is", "was", "are", "were", "did", "does", "do", "has", "have", "had",
    "will", "would", "the", "a", "an", "went", "finished", "came", "ranked",
)
# Possession/action verbs implying an ENTITY subject: "which [team] HAD the most".
# With a superlative, the answer is that entity's name (not the measure/id).
_POSSESSIVE_VERBS = (
    "had", "have", "has", "got", "scored", "won", "made", "played", "recorded",
    "went", "finished", "came", "ranked", "earned", "gained", "achieved",
)


def _expects_label(question: str) -> Optional[bool]:
    """True if the answer should be an ENTITY NAME (not a bare number)."""
    q = question.strip().lower()
    if re.search(r"\b(how many|how much|number of|what percentage|how often)\b", q):
        return False
    if re.match(r"^(when|how)\b", q):
        return False
    if re.search(r"\b(who|whom|whose)\b", q):
        return True
    if re.search(r"\bname (the|of|all)\b", q):
        return True
    m = re.search(r"\b(which|what)\s+([a-z]+)\b", q)
    if m:
        noun = m.group(2)
        if noun in _NUMERIC_NOUNS:
            return False
        # skip copulas/auxiliaries/determiners — those aren't the entity noun
        if noun in _COPULAS:
            # "which had the MOST/LEAST ..." → argmax over an entity: wants a name.
            # (e.g. "compare draws, which had the least points?" → the Artist, not '5')
            # Restrict to possession verbs: "which WAS the highest" may want the value.
            if noun in _POSSESSIVE_VERBS and _SUPERLATIVE_RE.search(q):
                return True
            return None
        return True  # "which monarch / what team ..." → wants an entity label
    return None


def answer_type_issues(question: str, items: list[str]) -> list[str]:
    """B-axis (high precision): a question that clearly wants a NAME was answered with a
    bare number. Catches argmax mistakes that return the measure instead of the label
    (e.g. 'which entry had the least points?' answered '5' instead of the row's name)."""
    if len(items) != 1:
        return []
    ans = items[0].strip()
    if not ans or not _is_number(ans):
        return []
    # a 4-digit value could be a legitimate year answer; don't flag those
    if re.fullmatch(r"\d{4}", ans):
        return []
    if _expects_label(question) is True:
        return [
            f"The question asks for a name/label but the answer '{ans}' is a bare number. "
            "If this is an argmax question ('which/who had the most/least X'), return the "
            "ROW LABEL (the name), not the value of X."
        ]
    return []


def _is_number(s: str) -> bool:
    try:
        float(s.replace(",", "").strip())
        return True
    except ValueError:
        return False


# Labels of summary/aggregation rows that are almost never a valid entity answer.
_SUMMARY_ANSWER_LABELS = {
    "total", "totals", "total:", "grand total", "sum", "subtotal", "average",
    "averages", "avg", "mean", "overall", "all", "n/a", "—", "-",
}


def summary_label_issues(question: str, items: list[str]) -> list[str]:
    """B/C-axis (high precision): the answer is a summary-row LABEL like 'Total'.
    These rows are aggregations, not real entities — returning one as e.g. the
    'last discipline' / 'which team' answer is almost always a mistake (often the
    result of not excluding the Total row before taking first/last/argmax)."""
    if len(items) != 1:
        return []
    ans = items[0].strip().lower()
    if ans in _SUMMARY_ANSWER_LABELS:
        return [
            f"The answer '{items[0]}' is a SUMMARY/total-row label, not a real data "
            "entity. Exclude summary rows (Total/Average/...) before taking the "
            "first/last/most/least, then return the actual data row."
        ]
    return []


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


def _norm_num(s: str) -> str:
    """Strip thousands separators and surrounding spaces for numeric comparison."""
    return re.sub(r"[,\s]", "", s.strip())


def format_drift_issues(items: list[str], df: "pd.DataFrame") -> list[str]:
    """A-axis (high precision): the answer is the SAME value as a real cell but with
    the FORMATTING stripped/changed. WTQ wants the cell's exact text.

    Catches e.g. answer '15.5' when a cell is '15.5%'; answer '1200' when the cell is
    '1,200'; answer 'USA' when a cell is 'USA*'. Conservative: only single-item answers,
    and only when a cell is exactly the answer plus a short unit/symbol wrapper.
    """
    if len(items) != 1:
        return []
    ans = items[0].strip()
    if not ans:
        return []
    ans_l = ans.lower()
    ans_num = _norm_num(ans)
    for col in df.columns:
        for val in df[col].astype(str).unique():
            v = val.strip()
            if not v or v.lower() == ans_l:
                return []  # exact cell match somewhere -> answer is already correct format
    candidates: set[str] = set()
    for col in df.columns:
        for val in df[col].astype(str).unique():
            v = val.strip()
            vl = v.lower()
            if not v or vl == ans_l:
                continue
            # (a) cell = answer + a trailing SYMBOL/unit. Restricted to high-precision
            # symbols (%, $, *, °) so we never mis-flag a count like '5' against '5th'.
            if vl.startswith(ans_l) and 0 < len(vl) - len(ans_l) <= 3:
                tail = v[len(ans):].strip()
                if tail and re.fullmatch(r"[%\$\*°]+", tail):
                    candidates.add(v)
            # (b) same number, different separators/decimals (1200 vs 1,200)
            elif ans_num and _norm_num(v) == ans_num and any(ch.isdigit() for ch in ans):
                candidates.add(v)
    if not candidates:
        return []
    sample = sorted(candidates)[:3]
    return [
        f"Your answer '{ans}' matches a table cell except for formatting (cells like "
        f"{sample}). WTQ answers copy the cell EXACTLY — keep the unit/symbol/separators "
        f"as shown (e.g. keep '%', keep '1,200'). Resubmit using the cell's exact text."
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
    # C-axis: list vs single. Strict-singular phrasing → flag even 2 items;
    # otherwise keep the looser >=3 guard to avoid false alarms.
    if _expects_exactly_one(question) and len(items) >= 2:
        issues.append(
            f"The question expects a SINGLE answer but {len(items)} items were returned: {items}. "
            "Pick the one the question asks for (e.g. the first/only/top match)."
        )
    elif _expects_single(question) is True and len(items) >= 3:
        issues.append(
            f"The question seems to expect a single answer but {len(items)} items were returned: {items}. "
            "Re-check whether the filter is too broad."
        )
    # NOTE: answer_type_issues ('wants a name but got a number') measured 0/6 on the
    # 1000-set — many 'what car / which ranking / which works number / 8 or 21' answers
    # are legitimately numbers — so it is intentionally NOT wired in (net-harmful).
    # B/C-axis: the answer is a summary-row label (Total/Average/...).
    issues += summary_label_issues(question, items)
    return issues


# --------------------------------------------------------------------------
# Tier 2 — table-aware LLM check on an independent model
# --------------------------------------------------------------------------

_LLM_SYSTEM = """You are an INDEPENDENT verifier for table question answering. You
are a different model from the one that produced the answer. The solver worked
directly with this table and is PRESUMED CORRECT. Your job is to catch clear,
objective mistakes — NOT to impose your own reading of an ambiguous question.

Check the answer along THREE axes:

A. FORM / PRECISION — Is each answer item copied EXACTLY as it appears in the
   relevant table cell? The gold answer almost always uses the TABLE'S OWN format. Flag if:
   - a value COPIED from a cell is rounded/reformatted differently from that cell (e.g.
     "15.5" when the cell says "15.5%", or an abbreviation expanded) — keep the cell's
     units, symbols, separators, and minimal format.
   - DO NOT demand thousands separators on a COMPUTED number. A count/sum/difference/
     average is gold-scored as PLAIN digits, and adding a comma ("2121" -> "2,121") makes
     it WRONG (the scorer parses "2,121" as text, not the number 2121). Only keep a comma
     if the value is copied verbatim from a cell that has one. Never flag a plain computed
     number as "missing a comma".
   - a unit/symbol is added or removed (%, $, *, °);
   - an abbreviation is expanded (e.g. "United States" when the cell says "USA");
   - a non-English title is TRANSLATED or romanized when the question asks for the title:
     return the ORIGINAL-script cell (e.g. the "Episode title" column), NOT a separate
     "Translation"/"Romanized" column, unless the question explicitly asks for that;
   - part of a multi-part cell is dropped (e.g. a name that also carries dates/qualifiers).
B. INTENT — Does the answer match what the question asks, read AGAINST THIS TABLE?
   Think about what each word means in the CONTEXT of this specific table, then check the
   answer fits that meaning:
   - COMPARATIVE / SUPERLATIVE DIRECTION. "higher/better/top rank" means a BETTER result —
     and you must work out what "better" maps to HERE. A Rank/Position of 1 is the TOP, so
     "ranked higher" = the SMALLEST rank number (or, if the leading rows' rank cell is
     blank because the table is sorted by result, the EARLIER row). For a points/score
     column, bigger = better. Flag the answer if the solver clearly went the WRONG
     direction (e.g. called rank 4 "higher" than rank 2, or picked the smallest points as
     "most"). Re-read the table to decide the direction before flagging.
   - ANSWER TYPE. A question wanting a name/label must not be answered with a number (and
     vice versa). ARGMAX questions ("which/who had the most/least X") want the ROW LABEL —
     the descriptive NAME, not a numeric id, not a code/reference (e.g. a catalogue id like
     "PHQ 4"), and not the measure value itself. "what is the highest/largest X" where X
     identifies an item usually wants the ITEM (the language/name), not the number.
   - CARDINALITY. A singular question ("who", "the only/first X", "which <singular noun>")
     must return exactly ONE item; a clearly plural/"all"/"two" question may return several.
C. COUNTING / AGGREGATION — For counts/sums/averages/extremes, is there an OBJECTIVE
   mistake — e.g. a Total/summary row wrongly included, or a clearly double-counted row?
   REVIEW the list of values the student matched (like checking scratch paper — scan, do
   not recompute). Flag ONLY when you can SEE a concrete problem in that printed list:
   - DUPLICATE ENTITY: the student counted ROWS for a recurring real-world entity
     (people, countries, teams) and the printed names visibly REPEAT (e.g. "John T.
     Jordan" twice) → the count should likely be DISTINCT. (Only for entities that recur;
     for EVENTS/occurrences — games, appearances, wins — repeats are real, do NOT flag.)
   - MISSED VARIANT: the student used an exact match and the column clearly contains the
     SAME item written differently (accent "Salomé", suffix "Salome, Op. 55") that was
     left out → the match was too strict.
   Quote the specific repeated/variant value as evidence. If you cannot point to a
   concrete duplicate or variant in the list, do NOT raise a counting concern.

═══ INTERPRETATION GUARD (read before flagging) ═══
This dataset uses NATURAL, EVERYDAY phrasing. Take the question at face value with
common-sense interpretation. You MUST NOT flag an answer merely because YOU would
interpret the question differently. In particular, these are the SOLVER's call, not
yours — do NOT flag them:
- "majority / most" → the option that occurs most often (plurality). Do NOT demand
  strictly more than 50%.
- "how many / total number of X" → for EVENTS/occurrences (games, appearances, wins)
  this is the COUNT OF ROWS; do NOT demand de-duplication. BUT for recurring ENTITIES
  (people, countries, teams) the gold often counts DISTINCT ones — so flag a row-count
  ONLY if you can SEE repeated entity names in the student's matched list (axis C).
- "next / previous / after / before X" → trust the solver's chosen axis. If the
  question says "listed"/"in the table" it is the adjacent ROW; if there is a
  Year/Date/Rank column it is the neighbour along THAT value. Do NOT flag a defensible
  axis choice — but DO flag if the solver clearly used row +/- 1 while the table is
  sorted so that the row neighbour is the wrong direction in time (e.g. descending
  years and "after" returned an earlier year).
- ordering, ties, inclusive/exclusive ranges when the question is silent → trust the solver.

ONLY flag when you are highly confident the answer is OBJECTIVELY wrong (wrong cell,
wrong column, a number that contradicts the visible table, a Total row included by
mistake, or a clear form/precision error). When in doubt, PASS.

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

VALID flaws to flag (a concrete, OBJECTIVE mechanical error in a specific step):
- Wrong column: "Step 1 filtered by column 'Winner' but the question asks about 'Runner-up'"
- Total row included: "Step 3 summed all rows including the 'Total' row, double-counting"
- Truncated cell: "The answer '4x400 m' is a substring of cell '4x400 m relay' — use the full cell"
- Type mismatch: "The answer is a number but the question clearly asks for a name/label"
- Wrong direction for a comparative/superlative: think about what "higher/better/most/
  longer" means in THIS table, then flag if the code went the wrong way — e.g. it treated
  the LARGER rank number as "higher rank" (rank 1 is the top), or took the smallest value
  as "most". Re-read the table to confirm the intended direction before flagging.
- Counting the wrong granularity — scan the student's printed matched list:
  * DUPLICATE ENTITY: a row count for a recurring entity (people/countries/teams) whose
    printed names visibly repeat ("John T. Jordan" twice) → likely needs DISTINCT. (Not
    for events/occurrences — games, appearances — where repeats are real.)
  * MISSED VARIANT: an exact `==` match that skipped the same item written differently
    (accent "Salomé", suffix "Salome, Op. 55"). Quote the repeated/variant value.
- Code bug: the code crashed, used the wrong variable, or its printed output contradicts the answer

═══ INTERPRETATION GUARD — DO NOT cross this line ═══
This dataset uses NATURAL, EVERYDAY phrasing. The student worked directly with the
table and OWNS the interpretation of ambiguous wording. You must NOT flag a step
merely because YOU would interpret the question differently. Specifically, these are
NOT flaws — treat them as the student's correct call:
- "majority / most" meaning the most frequent option (plurality), not strictly >50%.
- "how many / total number of X" meaning a COUNT OF ROWS for EVENTS/occurrences (games,
  appearances, wins) — do NOT demand de-dup there. (For recurring ENTITIES — people,
  countries, teams — only flag a row count if you SEE repeated names in the printed list.)
- "next / after / before" along whatever axis the student reasonably chose (listed row
  order, or a Year/Date/Rank value) — do NOT flag a defensible axis choice; only flag a
  clear direction error (e.g. a descending-year table where "after" returned an earlier year).
- choosing .count() vs .nunique(), >= vs >, inclusive vs exclusive when the question
  is SILENT — the student decides; do NOT flag.

INVALID — do NOT do these:
- Do NOT compute the answer yourself from scratch
- Do NOT flag a "wrong aggregation" or "missing de-dup" that is really just YOUR
  alternate reading of an ambiguous question (see the guard above)
- Do NOT say "the answer should be X" without pointing to a specific flawed step
- Do NOT invent flaws; if the student's work looks sound, say it passes
- Do NOT flag stylistic issues — only flag objective logical/factual errors

When in doubt, set "flawed": false. A false alarm that overturns a correct answer is
much worse than letting a borderline answer stand.

Respond with JSON only:
{
  "flawed": true|false,
  "step": "which step is flawed, e.g. 'run_python step 2' or 'final answer format'",
  "flaw": "ONE sentence naming the concrete OBJECTIVE error (the headline)",
  "reasoning": "2-4 sentences explaining the evidence: what you see in the table vs what the code does, why this is OBJECTIVELY wrong (not just a different interpretation)",
  "test": "one line of pandas code the student can run to verify this claim (or empty string if not applicable)"
}"""

_AUDIT_FOLLOWUP_SYSTEM = """You are a meticulous code auditor in round {round} of a debate about a table question-answering answer.

CONTEXT: In a previous round you raised a concern about the student's work. The student has since run
additional code to respond to your challenge. The code history now includes BOTH the original steps
AND their counter-code (the newer steps at the end).

Your task:
1. Review the student's counter-code carefully.
2. If their counter-code ADDRESSES your previous concern: set "flawed": false and "resolution": "accepted".
3. If their counter-code does NOT address the core issue, or reveals a NEW concrete flaw: flag it.
   - Be specific: point to the counter-code step that fails or the original flaw that persists.
   - Your previous concern was: {previous_concern}

BIAS TOWARD ACCEPTANCE: If the student ran code that is logically plausible and reaches a consistent
conclusion — even if you might have done it differently — accept their approach. Only reject if you
can point to a CONCRETE logical error in their code.

Respond with JSON only:
{{
  "flawed": true|false,
  "resolution": "accepted"|"rejected"|"new_issue",
  "step": "which step is still flawed (empty if accepted)",
  "flaw": "ONE sentence naming the remaining/new error (empty if accepted)",
  "reasoning": "2-4 sentences: why the counter-code does or does not resolve the concern, citing specific code lines or table values as evidence",
  "test": "one line of pandas code to verify the remaining claim (or empty string)"
}}"""


def audit_check(
    client: LLMClient,
    question: str,
    items: list[str],
    code_history: str,
    table_view: str = "",
    *,
    debate_round: int = 1,
    previous_concern: str = "",
    max_tokens: int = 1024,
) -> VerifyResult:
    """Audit the generator's own code history for a specific logical flaw.

    In round 1: fresh audit looking for any flaw.
    In round 2+: follow-up audit checking whether the generator's counter-code
    addressed the previous concern. Biased toward acceptance in later rounds.

    Returns the flaw description + a one-liner test the generator can run to
    verify the claim. Does NOT compute a new answer.
    """
    if debate_round <= 1:
        system = _AUDIT_SYSTEM
    else:
        system = _AUDIT_FOLLOWUP_SYSTEM.format(
            round=debate_round,
            previous_concern=previous_concern or "(not recorded)",
        )

    round_note = ""
    if debate_round > 1:
        round_note = (
            f"\n\n[Note: This is debate round {debate_round}. "
            f"The newer code steps at the end of the history are the student's response "
            f"to the previous concern: {previous_concern or '(see prior round)'}]"
        )

    user = (
        f"Question: {question}\n\n"
        f"Student's proposed answer: {items}\n\n"
        f"Table sample:\n{table_view or '(not provided)'}\n\n"
        f"Student's reasoning steps:\n{code_history or '(no code steps recorded)'}"
        f"{round_note}\n"
    )
    try:
        resp = client.chat(
            messages=[{"role": "system", "content": system},
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
    reasoning = str(parsed.get("reasoning") or "").strip()
    test_code = str(parsed.get("test") or "").strip()
    if not flaw_desc:
        return VerifyResult(ok=True, source="none", model=client.config.model, raw=raw[:300])

    issue = f"[{step_desc}] {flaw_desc}" if step_desc else flaw_desc
    return VerifyResult(
        ok=False, issues=[issue], source="audit",
        model=client.config.model,
        verifier_code=test_code,
        verifier_reasoning=reasoning,
        fix_hint=f"Run: {test_code}" if test_code else (reasoning[:120] if reasoning else ""),
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
    debate_round: int = 1,
    previous_concern: str = "",
) -> VerifyResult:
    """Three tiers, all running independently:
      1. deterministic rules (no model) — short-circuits on high-precision hits
      2. A/B/C dimensional review — checks answer quality (form, intent, counting)
      3. AUDIT of the generator's code history — finds specific logical flaws
         (does NOT re-answer; points to a concrete step and provides a test snippet)

    In debate_round >= 2, the audit is run as a follow-up (checking whether the
    generator's counter-code addressed the previous concern). Biased toward acceptance.
    """
    if not items:
        return VerifyResult(ok=True, source="none")

    # Tier 1: deterministic (no model cost) — only on round 1.
    # NOTE: cell_substring_issues / format_drift_issues were measured on a 1000-example
    # set to have LOW precision (substr 6/26, drift 0/1) — they mostly fire on CORRECT
    # answers (multi-line cells, 'India (IND)', month names) and fixed nothing, so they
    # are intentionally NOT wired in. WTQ's multi-line gold is too inconsistent to snap.
    if debate_round <= 1:
        det = deterministic_issues(question, items)
        if det:
            return VerifyResult(ok=False, issues=det, source="deterministic", axis="A")

    if client is None or not use_llm:
        return VerifyResult(ok=True, source="none")

    # Tier 2: A/B/C dimensional review (skip in follow-up rounds — focus on audit)
    review: VerifyResult
    if debate_round <= 1:
        review = llm_check(client, question, items, table_view, evidence_summary)
    else:
        review = VerifyResult(ok=True, source="none")

    # Tier 3: audit — round 1 is fresh; round 2+ checks counter-code
    audit = (
        audit_check(
            client, question, items, code_history, table_view,
            debate_round=debate_round,
            previous_concern=previous_concern,
        )
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
        verifier_code=audit.verifier_code,
        verifier_reasoning=audit.verifier_reasoning or review.fix_hint,
    )


_AXIS_HINT = {
    "A": "Form/precision: copy the value EXACTLY as the cell shows it (no rounding, "
         "no added/removed units, keep multi-part cells whole, don't expand abbreviations).",
    "B": "Intent: make sure the answer TYPE matches the question (name vs number, single vs list).",
    "C": "Counting: de-duplicate, skip blank/'–' cells, and handle any Total/summary row correctly.",
}


def build_debate_prompt(
    vr: VerifyResult,
    solver_answer: list[str],
    debate_round: int = 1,
    previous_concern: str = "",
) -> str:
    """Debate-mode prompt shown to the generator after the auditor flags a concern.

    Structure:
      1. Header — which round, which model, what answer is under scrutiny
      2. Claim headline — one-sentence flaw description
      3. Auditor's reasoning — natural-language evidence from the table / code
      4. Verification code — the one-liner pandas test the generator should run
      5. Instructions — maintain confidence, only change on code evidence
    """
    round_header = (
        "═══ AUDITOR REPORT ═══"
        if debate_round <= 1
        else f"═══ AUDITOR ROUND {debate_round} RESPONSE ═══"
    )
    lines = [round_header]

    if debate_round <= 1:
        lines.append(
            f"An independent auditor (model: {vr.model}) reviewed your reasoning steps "
            f"and raised a specific concern about your answer {solver_answer}."
        )
    else:
        lines += [
            f"The auditor (model: {vr.model}) reviewed your counter-code and still "
            f"has an unresolved concern about your answer {solver_answer}.",
            f"Previous concern: {previous_concern or '(see round 1)'}",
        ]

    lines.append("")

    # ── Claim headline ───────────────────────────────────────────────────────
    if vr.issues:
        lines.append("▶ CLAIM: " + "; ".join(vr.issues))
    if vr.axis and vr.axis in _AXIS_HINT:
        lines.append(f"  Category: Axis {vr.axis} — {_AXIS_HINT[vr.axis]}")

    # ── Natural-language reasoning ────────────────────────────────────────────
    if vr.verifier_reasoning:
        lines += [
            "",
            "▶ AUDITOR'S REASONING:",
            vr.verifier_reasoning,
        ]

    # ── Verification code ─────────────────────────────────────────────────────
    if vr.verifier_code:
        lines += [
            "",
            "▶ SUGGESTED TEST (run this to verify or refute the claim):",
            "```python",
            vr.verifier_code.strip(),
            "```",
        ]

    # ── Instructions ─────────────────────────────────────────────────────────
    lines += [
        "",
        f"═══ YOUR TURN — DEBATE ROUND {debate_round} (extra steps granted) ═══",
        "You are the expert who worked directly with this table.",
        "Default stance: your answer is PRESUMED CORRECT.",
        "",
        "FIRST, classify the auditor's claim:",
        "  (a) A CONCRETE MECHANICAL ERROR — wrong column, a number that contradicts",
        "      the table, a Total row included by mistake, a truncated cell, code that",
        "      crashed. These are objective and worth testing.",
        "  (b) A DIFFERENT INTERPRETATION of the question — e.g. 'majority means >50%',",
        "      'should be distinct count', 'should be chronological order'. The question's",
        "      meaning is YOUR call as the on-the-ground expert. Re-read the EXACT wording:",
        "        - 'majority/most' = the most frequent option (you do NOT need >50%).",
        "        - 'how many / total number' = count of matching ROWS (NOT distinct,",
        "          unless the question literally says 'distinct'/'different').",
        "        - 'next/after (listed)' = the adjacent ROW in table order (NOT chronological,",
        "          unless it says 'earliest'/'latest').",
        "",
        "DECISION RULE:",
        "  • If (b) interpretation dispute and the question wording supports YOUR reading →",
        "    KEEP your original answer. Resubmit it and briefly state the wording you relied on.",
        "    Computing a different quantity does NOT prove your answer wrong.",
        "  • If (a) mechanical error → run the test. If it CONFIRMS a real bug, fix it and",
        "    resubmit. If it REFUTES the claim, resubmit your ORIGINAL answer with the output.",
        "",
        "Do not abandon a correct answer just because the auditor computed a different number.",
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
