"""Compare two run JSONs (baseline vs agent) example-by-example.

Recomputes correctness with the official evaluator and buckets the confusion,
then prints the regression set (baseline right, agent wrong) which is where the
agentic approach is currently losing.

Run: python -m scripts.compare_runs <baseline.json> <agent.json>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, evaluator

SPLIT = "random-split-1-dev"


def load_rows(path):
    with open(path, encoding="utf8") as f:
        return {r["id"]: r for r in json.load(f)["rows"]}


def correct(targets, ex_id, items):
    if ex_id not in targets:
        return None
    pred_vals = evaluator.to_value_list(items) if items else []
    return evaluator.check_denotation(targets[ex_id], pred_vals)


def main() -> int:
    base = load_rows(sys.argv[1])
    agent = load_rows(sys.argv[2])
    tagged = evaluator.find_tagged_path(data.DEFAULT_DATASET_ROOT, SPLIT)
    targets = evaluator.load_targets_from_tagged(tagged)

    ids = [i for i in base if i in agent]
    buckets = {"both_right": [], "both_wrong": [], "agent_wins": [], "regress": []}
    for i in ids:
        b = correct(targets, i, base[i]["pred"])
        a = correct(targets, i, agent[i]["pred"])
        if b and a:
            buckets["both_right"].append(i)
        elif not b and not a:
            buckets["both_wrong"].append(i)
        elif a and not b:
            buckets["agent_wins"].append(i)
        else:
            buckets["regress"].append(i)

    print(f"n={len(ids)}")
    for k, v in buckets.items():
        print(f"  {k:12s} = {len(v)}")

    print("\n==== REGRESSIONS (baseline RIGHT, agent WRONG) ====")
    for i in buckets["regress"]:
        r = agent[i]
        print(f"\n[{i}] src={r.get('src')} steps={r.get('steps')}")
        print(f"  Q     : {r['q']}")
        print(f"  gold  : {r['gold']}")
        print(f"  base  : {base[i]['pred']}")
        print(f"  agent : {r['pred']}")

    print("\n==== AGENT WINS (baseline WRONG, agent RIGHT) ====")
    for i in buckets["agent_wins"]:
        r = agent[i]
        print(f"  [{i}] Q={r['q'][:70]} | gold={r['gold']} base={base[i]['pred']} agent={r['pred']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
