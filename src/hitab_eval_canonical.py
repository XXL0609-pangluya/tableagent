"""Convention-invariant ("canonical") scoring for HiTab.

`hitab_eval.py` is the STRICT scorer (kept untouched). This module adds a scorer
that is invariant to HiTab's *documented* gold-format conventions, so that a
prediction which is the same underlying quantity as gold — differing only by a
known convention transform — counts as correct. This lets us measure the real
(epistemic) error population without the aleatoric format-noise floor.

The governing principle we established from the gold audit (scripts/audit_hitab_gold.py):
gold = the VERBATIM output of the annotated formula. Surface wording ("percent")
is ambiguous, so the same quantity legitimately appears as 0.521 OR 52.1, as a
positive magnitude OR a signed cell read, with computed(6dp) vs display(1dp)
precision. We treat these as ONE canonical quantity.

Equivalences applied (each preserves the underlying quantity):
  - scale: pred == gold, or gold*100, or gold/100        (percent<->decimal)
  - sign : |pred| == |gold|                               (decline/change magnitude)
  - precision: relative tolerance (default 1e-2)          (display rounding)
  - list : split "a-b" / "a to b", dedupe, set-match
  - label: exact-norm, singular/plural, number-vs-number+unit word

`x1000` (unit/thousands) is OFF by default (genuinely ambiguous: convention vs a
real 1000x mistake); enable with allow_x1000=True to see its effect.

Everything is reported as BOTH strict and canonical so nothing is hidden.
"""
from __future__ import annotations

import re

from src.hitab_eval import _to_number, _norm_str, answer_match  # strict pieces


def _round_eq(a, b):
    """True iff a and b are equal up to display ROUNDING (not a loose relative
    tolerance). i.e. one side is the other rounded to k decimals for small k.
    So 1.5 == round(1.4966, 1) is accepted, but 435 vs 433 is NOT (no rounding
    of a full-precision value produces the other)."""
    if abs(a - b) <= 1e-6:
        return True
    for k in range(0, 7):
        if abs(round(b, k) - a) <= 1e-9 or abs(round(a, k) - b) <= 1e-9:
            return True
    return False


def _scale_eq(p, g, rel, allow_x1000):
    factors = [1.0, 100.0, 0.01]
    if allow_x1000:
        factors += [1000.0, 0.001]
    for f in factors:
        if _round_eq(p, g * f):
            return True
    return False


def _num_eq(p, g, rel, allow_x1000):
    """Convention-invariant numeric equality: scale (x100) and sign-invariant."""
    if _scale_eq(p, g, rel, allow_x1000):
        return True
    # sign-invariant (decline/change reported as positive magnitude)
    if _scale_eq(abs(p), abs(g), rel, allow_x1000):
        return True
    return False


def _depluralize(t):
    if t.endswith("es") and len(t) > 3:
        return t[:-2]
    if t.endswith("s") and len(t) > 2:
        return t[:-1]
    return t


def _lead_number(s):
    m = re.match(r"^[\s$]*[-−–]?\s*[\d,]*\.?\d+", str(s))
    return _to_number(m.group(0)) if m else None


def _item_eq(p, g, rel, allow_x1000):
    pn, gn = _to_number(p), _to_number(g)
    if pn is not None and gn is not None:
        return _num_eq(pn, gn, rel, allow_x1000)
    # string side
    np_, ng = _norm_str(p), _norm_str(g)
    if np_ == ng or _depluralize(np_) == _depluralize(ng):
        return True
    # number vs number+unit-word ("326" ~ "326 days")
    lp, lg = _lead_number(p), _lead_number(g)
    if pn is not None and lg is not None and _num_eq(pn, lg, rel, allow_x1000) and gn is None:
        return True
    if gn is not None and lp is not None and _num_eq(gn, lp, rel, allow_x1000) and pn is None:
        return True
    return False


def _split(items):
    out = []
    for x in items:
        parts = re.split(r"[-−–]|\bto\b", str(x))
        out.extend(p for p in parts if str(p).strip() != "")
    return out


def _clean(items):
    return [x for x in (items or []) if str(x).strip() != ""]


def _set_match(pred, gold, rel, allow_x1000):
    if len(pred) != len(gold):
        return False
    used = [False] * len(pred)
    for g in gold:
        hit = False
        for i, p in enumerate(pred):
            if not used[i] and _item_eq(p, g, rel, allow_x1000):
                used[i] = True
                hit = True
                break
        if not hit:
            return False
    return True


def canonical_match(pred_items, gold_items, rel=1e-2, allow_x1000=False):
    """True iff pred equals gold up to HiTab's documented format conventions."""
    pred, gold = _clean(pred_items), _clean(gold_items)
    if not gold:
        return False
    if _set_match(pred, gold, rel, allow_x1000):
        return True
    # dedupe (drop duplicate predictions) then match
    def dedupe(xs):
        seen, out = [], []
        for x in xs:
            k = _norm_str(x)
            if k not in seen:
                seen.append(k)
                out.append(x)
        return out
    if _set_match(dedupe(pred), gold, rel, allow_x1000):
        return True
    # split "a-b"/"a to b" ranges on either side
    if _set_match(_split(pred), gold, rel, allow_x1000):
        return True
    if _set_match(pred, _split(gold), rel, allow_x1000):
        return True
    return False


def score(results, rel=1e-2, allow_x1000=False):
    """results: list of {id, pred, gold, correct(strict, optional)}.
    Returns strict/canonical accuracy and the residual (canonical-wrong) ids."""
    n = len(results)
    strict = canon = 0
    recovered, residual = [], []
    for r in results:
        s = answer_match(_clean(r.get("pred")), _clean(r.get("gold")))
        c = s or canonical_match(r.get("pred"), r.get("gold"), rel, allow_x1000)
        strict += int(s)
        canon += int(c)
        if c and not s:
            recovered.append(r.get("id"))
        if not c:
            residual.append(r.get("id"))
    return {
        "n": n,
        "strict_correct": strict, "strict_acc": round(strict / n, 4) if n else 0.0,
        "canonical_correct": canon, "canonical_acc": round(canon / n, 4) if n else 0.0,
        "recovered_by_convention": len(recovered),
        "residual_real_errors": len(residual),
        "recovered_ids": recovered, "residual_ids": residual,
    }
