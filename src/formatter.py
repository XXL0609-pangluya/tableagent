"""Answer formatting: model output / structured result -> list of answer items.

The evaluator does set matching with its own normalization, so here we only need
to split a raw answer into the right number of items and strip obvious wrappers.
List answers in WTQ are separated by '|'.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _strip_item(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def parse_answer_text(text: str) -> list[str]:
    """Parse a model's free-form answer into answer items.

    Accepts: a JSON array, a '|'-separated list, or a single value. Falls back to
    the last non-empty line if the model added prose.
    """
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []

    # Model emitted a textual tool call like submit_answer(["3"]) / submit_answer(items=[...]).
    m = re.search(r"submit_answer\(\s*(?:items\s*=\s*)?(\[.*\]|\"[^\"]*\"|'[^']*'|[^)]*)\)", text, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        try:
            arr = json.loads(inner)
            if isinstance(arr, list):
                return [_strip_item(str(x)) for x in arr if str(x).strip()]
            return [_strip_item(str(arr))]
        except json.JSONDecodeError:
            return [_strip_item(inner)]

    # JSON array?
    if text.startswith("["):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [_strip_item(str(x)) for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass

    # Strip a common "Answer:" prefix and take the last non-empty line.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[-1]
    for prefix in ("answer:", "Answer:", "ANSWER:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    parts = [p for p in (_strip_item(p) for p in text.split("|")) if p]
    return parts if parts else [_strip_item(text)]


_YESNO_STARTS = (
    "is ", "are ", "was ", "were ", "does ", "do ", "did ", "has ", "have ",
    "had ", "can ", "could ", "will ", "would ", "should ",
)


def _looks_yesno(question: str) -> bool:
    q = (question or "").strip().lower()
    return q.startswith(_YESNO_STARTS)


def _strip_count_unit(s: str, question: str) -> str:
    """For count questions, drop a trailing unit that just echoes the question's
    own noun: "what number of acts" + "5 acts" -> "5".

    High precision: only fires when the trailing word matches the noun right after
    'how many' / 'number of' in the question (so it won't touch gold-style units
    like "2 years" for a "how long" question)."""
    m = re.search(r"\b(?:how many|number of)\s+([a-z]+)", question.lower())
    if not m:
        return s
    noun = m.group(1).rstrip("s")
    m2 = re.match(r"^(\d[\d,]*)\s+([A-Za-z]+)$", s.strip())
    if m2 and m2.group(2).lower().rstrip("s") == noun:
        return m2.group(1)
    return s


def _strip_venue_prefix(s: str, question: str) -> str:
    """Drop an away-game venue marker: "at New York Titans" -> "New York Titans".

    Gated on opponent/team/who-style questions so it only applies to schedule
    tables where 'at' marks an away game, not part of the name."""
    ql = question.lower()
    if not re.search(r"\b(opponent|team|who|which|whom|play(?:ed)?|beat|defeat|lose|lost|tie|won|win)\b", ql):
        return s
    m = re.match(r"^(?:at|vs\.?|@)\s+([A-Z].*)$", s.strip())
    return m.group(1).strip() if m else s


def normalize_items(items: list[str], question: str = "") -> list[str]:
    """Light, lossless-ish answer cleanup before grading.

    The evaluator already handles case/commas/parentheses, so we only fix surface
    issues the model commonly emits that the evaluator does NOT forgive:
      - python/yes-no boolean phrasing matched to what the question expects:
          * "true or false?" questions  -> true/false
          * plain yes/no questions       -> yes/no
      - a leading row-number marker like '#163' -> '163'
      - collapse an accidental multi-line cell to its first line
    """
    out: list[str] = []
    ql = (question or "").lower()
    wants_truefalse = "true or false" in ql or "true/false" in ql
    yesno = _looks_yesno(question)
    bool_map = {True: "true", False: "false"} if wants_truefalse else {True: "yes", False: "no"}
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        low = s.lower()
        if low in ("true", "yes"):
            s = bool_map[True]
        elif low in ("false", "no"):
            s = bool_map[False]
        elif yesno and low in ("yes.", "no."):
            s = low[:-1]
        if "\n" in s:
            s = s.split("\n", 1)[0].strip()
        if len(s) > 1 and s[0] == "#" and s[1:].lstrip().isdigit():
            s = s[1:].strip()
        s = _strip_count_unit(s, question)
        s = _strip_venue_prefix(s, question)
        if s:
            out.append(s)
    return out


def items_from_structured(structured: dict[str, Any]) -> list[str]:
    """Coerce run_python's `answer_items` into a clean list[str]."""
    items = structured.get("answer_items")
    if items is None:
        return []
    if not isinstance(items, (list, tuple)):
        items = [items]
    return [str(x).strip() for x in items if str(x).strip()]
