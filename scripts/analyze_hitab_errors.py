"""Separate HiTab errors into FORM/CONVENTION noise vs REAL reasoning errors.

Given a run's preds.json (id, q, pred, gold, correct, ...), classify every wrong
example into one of:

  - crash        : the example errored out (no pred).
  - sign_only    : numeric, |pred| matches |gold| but the sign differs.
  - scale_x100   : numeric, pred == gold*100 or gold/100 (decimal<->percent).
  - rounding     : numeric one-to-one match within a LOOSE tol but not the strict
                   1e-4 the scorer uses (pure precision / early rounding).
  - list_format  : same underlying values but differ by dedup / hyphen-split /
                   count (multiset/format mismatch, not a value mistake).
  - real_error   : none of the above -> a genuine wrong cell / computation / label.

FORM buckets (sign_only, scale_x100, rounding, list_format) are aleatoric /
output-contract noise; real_error is the clean population for trigger/adjudication
research. Writes a categorized JSON and a short markdown summary next to the preds.

Usage:
  .venv/bin/python scripts/analyze_hitab_errors.py <preds.json> [more_preds.json ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hitab_eval import _to_number, _norm_str, answer_match  # noqa: E402


def _nums(items):
    """Return list of floats if EVERY non-blank item is numeric, else None."""
    vals = []
    for x in (items or []):
        if str(x).strip() == "":
            continue
        n = _to_number(x)
        if n is None:
            return None
        vals.append(n)
    return vals


def _close(a, b, rel=1e-4, abs_=1e-6):
    if abs(a - b) <= abs_:
        return True
    return abs(a - b) / max(abs(a), abs(b), 1e-9) <= rel


def _one_to_one(pred, gold, ok):
    """Is there a one-to-one matching where each pair satisfies ok(p, g)?"""
    if len(pred) != len(gold):
        return False
    used = [False] * len(pred)
    for g in gold:
        hit = False
        for i, p in enumerate(pred):
            if not used[i] and ok(p, g):
                used[i] = True
                hit = True
                break
        if not hit:
            return False
    return True


def _split_hyphens(items):
    out = []
    for x in items:
        parts = re.split(r"[-−–]|\bto\b", str(x))
        out.extend(p for p in parts if p.strip() != "")
    return out


def _depluralize(t):
    if t.endswith("es") and len(t) > 3:
        return t[:-2]
    if t.endswith("s") and len(t) > 2:
        return t[:-1]
    return t


def _lead_number(s):
    m = re.match(r"^[\s$]*[-−–]?\s*[\d,]*\.?\d+", str(s))
    return _to_number(m.group(0)) if m else None


def _soft_eq(p, g):
    """Convention-level string equality: exact-norm, singular/plural, or
    number-vs-number+unit (e.g. '326' ~ '326 days')."""
    np_, ng = _norm_str(p), _norm_str(g)
    if np_ == ng:
        return True
    if _depluralize(np_) == _depluralize(ng):
        return True
    pnum, gnum = _to_number(p), _to_number(g)
    lp, lg = _lead_number(p), _lead_number(g)
    # one side is a bare number, the other is that same number + a unit word
    if pnum is not None and lg is not None and _close(pnum, lg) and _to_number(g) is None:
        return True
    if gnum is not None and lp is not None and _close(gnum, lp) and _to_number(p) is None:
        return True
    return False


def _has_containment(pred, gold):
    for p in pred:
        for g in gold:
            np_, ng = _norm_str(p), _norm_str(g)
            if np_ != ng and (np_ in ng or ng in np_):
                return True
    return False


def classify(pred, gold):
    pred = [p for p in (pred or []) if str(p).strip() != ""]
    gold = [g for g in (gold or []) if str(g).strip() != ""]
    pn, gn = _nums(pred), _nums(gold)

    # sign_only: same magnitudes, different signs (and not already matching)
    if pn is not None and gn is not None and len(pn) == len(gn):
        if _one_to_one(pn, gn, lambda p, g: _close(abs(p), abs(g))) and \
           not _one_to_one(pn, gn, lambda p, g: _close(p, g)):
            return "sign_only"
        # scale_x100: each pred is gold*100 or gold/100 (decimal<->percent)
        if _one_to_one(pn, gn, lambda p, g: (g != 0 and (_close(p, g * 100, rel=2e-2) or _close(p, g / 100, rel=2e-2)))):
            return "scale_x100"
        # scale_x1000: unit / thousands / verbatim-cell convention (ambiguous)
        if _one_to_one(pn, gn, lambda p, g: (g != 0 and (_close(p, g * 1000, rel=2e-2) or _close(p, g / 1000, rel=2e-2)))):
            return "scale_x1000"
        # rounding: loose match (rel 2e-2) but strict scorer fails
        if _one_to_one(pn, gn, lambda p, g: _close(p, g, rel=2e-2)):
            return "rounding"

    # list_format: same values ignoring dedup / hyphen-split / order
    def as_set(xs):
        return sorted(_norm_str(x) for x in xs)
    if as_set(set(pred)) == as_set(set(gold)) and as_set(pred) != as_set(gold):
        return "list_format"  # duplicates only
    if as_set(_split_hyphens(pred)) == as_set(gold) or as_set(pred) == as_set(_split_hyphens(gold)):
        return "list_format"  # hyphen split/join
    # numeric multiset equal but count differs (sub/superset with same values)
    if pn is not None and gn is not None:
        def close_in(v, pool):
            return any(_close(v, x) for x in pool)
        if all(close_in(g, pn) for g in gn) and all(close_in(p, gn) for p in pn) and len(pn) != len(gn):
            return "list_format"

    # label_form: one-to-one convention-level string equality (plural, unit word)
    if _one_to_one(pred, gold, _soft_eq):
        return "label_form"

    # borderline: same count with substring/containment overlap -> likely a
    # label-granularity issue but ambiguous (could be a wrong selection). Flag
    # separately so a human can eyeball rather than silently counting either way.
    if len(pred) == len(gold) and _has_containment(pred, gold):
        return "borderline_label"

    return "real_error"


def analyze(preds_path: str):
    data = json.load(open(preds_path))
    results = data.get("results", data if isinstance(data, list) else [])
    buckets: dict[str, list] = {k: [] for k in
                                ["crash", "sign_only", "scale_x100", "scale_x1000",
                                 "rounding", "list_format", "label_form",
                                 "borderline_label", "real_error"]}
    n = len(results)
    n_correct = 0
    for r in results:
        if r.get("correct"):
            n_correct += 1
            continue
        if "error" in r and "pred" not in r:
            buckets["crash"].append({"id": r.get("id"), "error": r.get("error")})
            continue
        cat = classify(r.get("pred"), r.get("gold"))
        buckets[cat].append({"id": r.get("id"), "q": r.get("q"),
                             "pred": r.get("pred"), "gold": r.get("gold")})

    n_wrong = n - n_correct
    form = sum(len(buckets[k]) for k in
               ["sign_only", "scale_x100", "rounding", "list_format", "label_form"])
    out = {
        "preds": preds_path,
        "n": n, "correct": n_correct, "wrong": n_wrong,
        "accuracy": round(n_correct / n, 4) if n else 0.0,
        "counts": {k: len(v) for k, v in buckets.items()},
        "form_noise_total": form,
        "real_error_total": len(buckets["real_error"]),
        "form_robust_accuracy": round((n_correct + form) / n, 4) if n else 0.0,
        "buckets": buckets,
    }
    base = Path(preds_path).with_name(Path(preds_path).stem + "_error_analysis")
    json.dump(out, open(str(base) + ".json", "w"), indent=2, ensure_ascii=False)
    json.dump([b["id"] for b in buckets["real_error"]],
              open(str(base.with_name("real_error_ids")) + ".json", "w"), indent=2)

    print(f"\n=== {preds_path} ===")
    print(f"n={n}  correct={n_correct} ({out['accuracy']*100:.1f}%)  wrong={n_wrong}")
    for k in ["sign_only", "scale_x100", "scale_x1000", "rounding", "list_format",
              "label_form", "borderline_label", "real_error", "crash"]:
        print(f"  {k:16s}: {len(buckets[k])}")
    print(f"  -> FORM/convention noise : {form}  ({form/max(n_wrong,1)*100:.0f}% of errors)")
    print(f"  -> unit_scale x1000 (ambig): {len(buckets['scale_x1000'])}")
    print(f"  -> borderline (eyeball)  : {len(buckets['borderline_label'])}")
    print(f"  -> REAL reasoning errors : {len(buckets['real_error'])}  ({len(buckets['real_error'])/max(n_wrong,1)*100:.0f}% of errors)")
    print(f"  form-robust accuracy (if all form noise were counted correct): {out['form_robust_accuracy']*100:.1f}%")
    print(f"  wrote {base}.json  and  real_error_ids.json")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        analyze(p)
