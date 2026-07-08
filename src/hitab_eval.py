"""HiTab answer matching (HiTab-specific, kept separate from evaluator.py).

HiTab gold answers are lists of numbers (mostly floats) or short strings. The
official metric is exact match after normalization, with numeric answers compared
as numbers. We implement a faithful, dependency-free matcher:

  - numbers: parsed (strip commas/%/$/spaces) and compared with a small tolerance
    (relative + absolute) so 154983 == 154983.0 and 30.999 ~= 31.0 style rounding.
  - strings: lowercased, stripped, punctuation-insensitive.
  - a prediction matches gold iff both are the same size and there is a
    one-to-one match (set semantics, like WTQ denotation).

WTQ's evaluator.py is left untouched.
"""
from __future__ import annotations

import re
from typing import Optional

_NUM_RE = re.compile(r"^[\s$]*[-−–]?\s*[\d,]*\.?\d+\s*%?\s*$")


def _to_number(s: str) -> Optional[float]:
    t = str(s).strip().replace(",", "").replace("−", "-").replace("–", "-")
    t = t.replace("$", "").replace("%", "").strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _norm_str(s: str) -> str:
    t = str(s).strip().lower()
    t = re.sub(r"[\s]+", " ", t)
    # drop surrounding quotes and trailing punctuation
    t = t.strip("'\"").strip()
    t = re.sub(r"[.,;:]+$", "", t)
    return t


def item_match(pred: str, gold: str, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> bool:
    """Match a single predicted item to a single gold item."""
    gp = _to_number(gold)
    pp = _to_number(pred)
    if gp is not None and pp is not None:
        if abs(gp - pp) <= abs_tol:
            return True
        denom = max(abs(gp), abs(pp), 1e-9)
        return abs(gp - pp) / denom <= rel_tol
    # fall back to string comparison
    return _norm_str(pred) == _norm_str(gold)


def answer_match(pred_items: list[str], gold_items: list[str]) -> bool:
    """Set-semantics match: same size and a one-to-one item correspondence."""
    pred = [p for p in (pred_items or []) if str(p).strip() != ""]
    gold = [g for g in (gold_items or []) if str(g).strip() != ""]
    if len(pred) != len(gold):
        return False
    used = [False] * len(pred)
    for g in gold:
        found = False
        for i, p in enumerate(pred):
            if not used[i] and item_match(p, g):
                used[i] = True
                found = True
                break
        if not found:
            return False
    return True


def evaluate_hitab(predictions: dict[str, list[str]], targets: dict[str, list[str]]) -> dict:
    """Score predictions (id -> items) against targets (id -> gold items)."""
    n = 0
    correct = 0
    per_example: dict[str, bool] = {}
    for ex_id, gold in targets.items():
        n += 1
        pred = predictions.get(ex_id)
        ok = answer_match(pred, gold) if pred is not None else False
        per_example[ex_id] = ok
        correct += int(ok)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "per_example": per_example,
    }
