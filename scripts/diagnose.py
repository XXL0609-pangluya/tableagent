"""Deep-dive specific example ids: table + gold vs pred + full trajectory.

Uses src/analysis.py formatting. Writes trace to results/trace_diagnose.jsonl
(separate from batch runs).

Run:
  python -m scripts.diagnose nt-1702 nt-4677 ...
  python -m scripts.diagnose --from-compare baseline.json agent.json regress
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis, data
from src.agent import load_prompts, run_example
from src.config import load_llm_config
from src.llm import LLMClient
from src.schemas import Budget
from src.tools.wtq_tools import build_registry
from src.trace import Tracer

SPLIT = "random-split-1-dev"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _ids_from_compare(base_path: str, agent_path: str, bucket: str) -> list[str]:
    base = analysis.load_run_rows(base_path)
    agent = analysis.load_run_rows(agent_path)
    return analysis.compare_buckets(base, agent).get(bucket, [])


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("usage:")
        print("  python -m scripts.diagnose <id> [<id> ...]")
        print("  python -m scripts.diagnose --from-compare <baseline.json> <agent.json> [bucket]")
        return 2

    if argv[0] == "--from-compare":
        if len(argv) < 3:
            print("usage: python -m scripts.diagnose --from-compare base.json agent.json [regress]")
            return 2
        bucket = argv[3] if len(argv) > 3 else "regress"
        ids = _ids_from_compare(argv[1], argv[2], bucket)
        print(f"# bucket={bucket} n={len(ids)} ids={ids[:5]}{'...' if len(ids)>5 else ''}\n")
    else:
        ids = argv

    byid = {e.id: e for e in data.load_examples(SPLIT) if e.id in set(ids)}
    cfg = load_llm_config()
    client = LLMClient(cfg)
    prompts = load_prompts()
    reg = build_registry()
    budget = Budget(max_steps=8)
    trace_path = os.path.join(RESULTS_DIR, "trace_diagnose.jsonl")
    if os.path.exists(trace_path):
        os.remove(trace_path)

    for ex_id in ids:
        ex = byid.get(ex_id)
        if ex is None:
            print(f"[{ex_id}] not found in {SPLIT}")
            continue
        tc = data.load_table(ex.table_path)
        tracer = Tracer(trace_path, ex.id)
        pred = run_example(ex, tc, reg, client, prompts, budget=budget, tracer=tracer)
        tracer.flush(extra={"question": ex.utterance, "gold": ex.target_value, "pred": pred.items})
        print(analysis.format_example_report(ex, tc, pred, tracer.events))

    print(f"(trace saved to {trace_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
