"""Candidate-pool diversity & selection-recoverability audit.

Prerequisite check for the "decision-right allocation" experiments: offline
replay of *who selects the final answer* only has signal where the candidate
pool actually contains >1 semantically-distinct answer. This script quantifies
that, using the OFFICIAL evaluator normalization (to_value_list + check_denotation)
so that '12467' vs '12,467' are NOT counted as a real disagreement.

Reports, segmented by whether debate fired (verify_retries > 0):
  - pool-size distribution
  - fraction of rows with >1 semantically-distinct candidate
  - ORACLE-SELECTOR upper bound: rows where the current final pred is WRONG but
    some candidate in the pool matches gold (i.e. a better selector could recover it)
  - current final accuracy vs oracle-selector accuracy (the ceiling for pure selection)

Usage:
    python3 scripts/candidate_diversity.py results/agent_qwen3.6-35b-a3b_mine_1000.json [more.json ...]
"""
from __future__ import annotations

import sys
import json
from collections import Counter

sys.path.insert(0, ".")
from src.evaluator import to_value_list, check_denotation  # noqa: E402


def load_rows(path: str) -> list[dict]:
    d = json.load(open(path))
    if isinstance(d, list):
        return d
    for key in ("rows", "results"):
        if isinstance(d.get(key), list):
            return d[key]
    raise ValueError(f"{path}: no rows/results list found")


def matches(a: list[str], b: list[str]) -> bool:
    """Official denotation match between two answer item-lists."""
    try:
        return check_denotation(to_value_list(a or []), to_value_list(b or []))
    except Exception:
        return (a or []) == (b or [])


def semantic_dedup(cands: list[list[str]]) -> list[list[str]]:
    """Collapse candidates that are equal under evaluator normalization."""
    uniq: list[list[str]] = []
    for c in cands:
        c = list(c) if isinstance(c, (list, tuple)) else [c]
        if not any(matches(c, u) for u in uniq):
            uniq.append(c)
    return uniq


def gold_of(r: dict) -> list[str]:
    g = r.get("gold")
    if isinstance(g, str):
        return [g]
    return list(g or [])


def pred_of(r: dict) -> list[str]:
    p = r.get("pred")
    if isinstance(p, str):
        return [p]
    return list(p or [])


def audit(path: str) -> None:
    rows = load_rows(path)
    seg = {"no_debate": [], "debate": []}
    for r in rows:
        key = "debate" if (r.get("verify_retries") or 0) > 0 else "no_debate"
        # debate result files lack verify_retries; infer from candidates/trace length
        if "verify_retries" not in r:
            cands = r.get("candidates")
            key = "debate" if (cands and len(cands) > 1) else "no_debate"
        seg[key].append(r)

    print(f"\n{'='*70}\n{path}\n  total rows: {len(rows)}")
    total_correct = 0
    total_oracle = 0
    total_recoverable = 0

    for name, rs in [("no_debate", seg["no_debate"]), ("debate", seg["debate"]), ("ALL", rows)]:
        if name == "ALL":
            rs = rows
        sizes = Counter()
        multi = 0
        recoverable = 0
        cur_correct = 0
        oracle_correct = 0
        for r in rs:
            cands = r.get("candidates") or []
            if not cands and r.get("pred") is not None:
                cands = [pred_of(r)]
            uniq = semantic_dedup(cands)
            sizes[len(uniq)] += 1
            if len(uniq) > 1:
                multi += 1
            gold = gold_of(r)
            pred = pred_of(r)
            cur_ok = matches(pred, gold)
            oracle_ok = any(matches(c, gold) for c in cands)
            cur_correct += cur_ok
            oracle_correct += oracle_ok
            if oracle_ok and not cur_ok:
                recoverable += 1
        n = max(1, len(rs))
        print(f"\n  [{name}] n={len(rs)}")
        print(f"    distinct-pool-size dist: {dict(sorted(sizes.items()))}")
        print(f"    rows with >1 distinct candidate: {multi} ({100*multi/n:.1f}%)")
        print(f"    current final acc:      {cur_correct}/{len(rs)} = {100*cur_correct/n:.1f}%")
        print(f"    ORACLE-selector acc:    {oracle_correct}/{len(rs)} = {100*oracle_correct/n:.1f}%")
        print(f"    selection-recoverable:  {recoverable} "
              f"(+{100*recoverable/n:.1f}pt ceiling from pure selection)")
        if name == "ALL":
            total_correct, total_oracle, total_recoverable = cur_correct, oracle_correct, recoverable

    print(f"\n  >>> Pure-selection headroom on this file: "
          f"{total_correct} -> {total_oracle} (+{total_recoverable})")


if __name__ == "__main__":
    paths = sys.argv[1:] or ["results/agent_qwen3.6-35b-a3b_mine_1000.json"]
    for p in paths:
        try:
            audit(p)
        except Exception as e:
            print(f"\n{p}: SKIP ({e})")
