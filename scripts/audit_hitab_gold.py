"""Audit HiTab GOLD answer-format consistency (dataset property, model-independent).

Goal: decide whether HiTab's gold obeys a *uniform per-question-type* format
standard, or whether the same operation/phrasing yields inconsistent gold forms
(e.g. a ratio sometimes 0.52, sometimes 52.1). We group by the dataset's own
`aggregation` label (the objective operation type) crossed with question-surface
cues ("percent" / "times" / "proportion" / decline/change), and characterize the
gold numeric form in each cell:

  - magnitude band: [0,1] decimal, (1,100] percent-scale, >100
  - sign: negative present?
  - precision: number of decimal places actually carried

Then it flags SURFACE inconsistency: cells where a single question cue maps to
more than one gold format, which is the aleatoric noise we must standardize away.

Usage:
  .venv/bin/python scripts/audit_hitab_gold.py [datasets/HiTab/data/test_samples.jsonl ...]
Writes <stem>_gold_audit.json next to the first input.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _agg(r):
    a = r.get("aggregation")
    return tuple(a) if isinstance(a, list) else (a,)


def _nums(ans):
    out = []
    for x in ans or []:
        s = str(x).strip().replace(",", "").replace("%", "").replace("−", "-")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def _decimals(x):
    s = repr(float(x))
    if "e" in s or "E" in s:
        return None
    return len(s.split(".")[1]) if "." in s else 0


def _band(v):
    a = abs(v)
    if a <= 1:
        return "[0,1]"
    if a <= 100:
        return "(1,100]"
    return ">100"


def _cue(q):
    ql = (q or "").lower()
    cues = []
    if "percent" in ql or "%" in ql:
        cues.append("percent")
    if "proportion" in ql or "share" in ql:
        cues.append("proportion")
    if re.search(r"\btimes\b|more likely|less likely", ql):
        cues.append("times/likely")
    if re.search(r"\bdeclin|\bdecreas|\bdrop|\bfell\b|\bloss\b|\bfewer\b|\breduction", ql):
        cues.append("decline")
    if re.search(r"\bchang|\bincreas|\bgrew\b|\bgrowth\b|\brose\b", ql):
        cues.append("change/increase")
    return cues or ["(other)"]


def audit(path: str):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)

    # 1. per-aggregation gold numeric form
    by_agg = defaultdict(lambda: {"n": 0, "numeric": 0, "bands": Counter(),
                                  "neg": 0, "dec_hist": Counter(), "multi": 0})
    for r in rows:
        a = _agg(r)
        d = by_agg[a]
        d["n"] += 1
        ns = _nums(r.get("answer"))
        if len(r.get("answer") or []) > 1:
            d["multi"] += 1
        if not ns:
            continue
        d["numeric"] += 1
        for v in ns:
            d["bands"][_band(v)] += 1
            if v < 0:
                d["neg"] += 1
            dc = _decimals(v)
            if dc is not None:
                d["dec_hist"][min(dc, 7)] += 1

    # 2. surface inconsistency: cue x aggregation -> band mix
    cue_band = defaultdict(Counter)          # cue -> band counts (single-number Qs)
    cue_agg_band = defaultdict(Counter)      # (cue) -> "agg|band"
    for r in rows:
        ns = _nums(r.get("answer"))
        if len(ns) != 1:
            continue
        b = _band(ns[0])
        a = ",".join(_agg(r))
        for c in _cue(r["question"]):
            cue_band[c][b] += 1
            cue_agg_band[c][f"{a} -> {b}"] += 1

    report = {"path": path, "n": n, "by_aggregation": {}, "cue_band": {}, "cue_agg_detail": {}}
    print(f"\n================ GOLD AUDIT: {path}  (n={n}) ================")
    print("\n--- per aggregation: gold numeric form ---")
    for a, d in sorted(by_agg.items(), key=lambda kv: -kv[1]["n"]):
        dec = ",".join(f"{k}dp:{v}" for k, v in sorted(d["dec_hist"].items()))
        bands = dict(d["bands"])
        print(f"  {str(a):28} n={d['n']:4}  numeric={d['numeric']:4}  "
              f"neg={d['neg']:3}  multi={d['multi']:3}  bands={bands}")
        print(f"      decimals: {dec}")
        report["by_aggregation"][",".join(a)] = {
            "n": d["n"], "numeric": d["numeric"], "neg": d["neg"],
            "multi": d["multi"], "bands": bands, "decimals": dict(d["dec_hist"])}

    print("\n--- surface cue -> gold magnitude band (single-number Qs) ---")
    for c, bc in sorted(cue_band.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(bc.values())
        mixed = "  <== MIXED" if len([k for k, v in bc.items() if v]) > 1 else ""
        print(f"  cue={c:16} total={tot:4}  bands={dict(bc)}{mixed}")
        report["cue_band"][c] = dict(bc)
        report["cue_agg_detail"][c] = dict(cue_agg_band[c])

    print("\n--- WHERE THE INCONSISTENCY LIVES (cue=percent, by aggregation->band) ---")
    for k, v in sorted(cue_agg_band["percent"].items()):
        print(f"    {k}: {v}")

    out = Path(path).with_name(Path(path).stem + "_gold_audit.json")
    json.dump(report, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")
    return report


if __name__ == "__main__":
    paths = sys.argv[1:] or ["datasets/HiTab/data/test_samples.jsonl"]
    for p in paths:
        audit(p)
