"""Print trajectories for a confusion bucket using saved batch traces.

Run: python -m scripts.inspect_traces <baseline.json> <agent.json> <trace.jsonl> [bucket]
  bucket in {regress, both_wrong, agent_wins}  (default: regress)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: python -m scripts.inspect_traces <baseline.json> <agent.json> <trace.jsonl> [bucket]")
        return 2
    base = analysis.load_run_rows(sys.argv[1])
    agent = analysis.load_run_rows(sys.argv[2])
    traces = analysis.load_traces_jsonl(sys.argv[3])
    bucket = sys.argv[4] if len(sys.argv) > 4 else "regress"
    print(analysis.format_bucket_from_runs(base, agent, traces, bucket))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
