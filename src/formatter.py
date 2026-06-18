"""Answer formatting: model output / structured result -> list of answer items.

The evaluator does set matching with its own normalization, so here we only need
to split a raw answer into the right number of items and strip obvious wrappers.
List answers in WTQ are separated by '|'.
"""
from __future__ import annotations

import json
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


def items_from_structured(structured: dict[str, Any]) -> list[str]:
    """Coerce run_python's `answer_items` into a clean list[str]."""
    items = structured.get("answer_items")
    if items is None:
        return []
    if not isinstance(items, (list, tuple)):
        items = [items]
    return [str(x).strip() for x in items if str(x).strip()]
