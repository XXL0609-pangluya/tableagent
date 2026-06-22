"""Compare two run JSONs (baseline vs agent) example-by-example.

Run: python -m scripts.compare_runs <baseline.json> <agent.json>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python -m scripts.compare_runs <baseline.json> <agent.json>")
        return 2
    base = analysis.load_run_rows(sys.argv[1])
    agent = analysis.load_run_rows(sys.argv[2])
    buckets = analysis.compare_buckets(base, agent)

    print(f"n={len([i for i in base if i in agent])}")
    for k, v in buckets.items():
        print(f"  {k:12s} = {len(v)}")

    print("\n==== REGRESSIONS (baseline RIGHT, agent WRONG) ====")
    for i in buckets["regress"]:
        r = agent[i]
        print(f"\n[{i}] src={r.get('src')} steps={r.get('steps')} verify={r.get('verify')}")
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
