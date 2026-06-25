"""Re-run only the examples that hit an LLM/API error (e.g. 429 rate-limit) in a
previous batch run, then merge the fresh predictions back into the results JSON and
re-score.

Run: python -m scripts.rerun_errors [which] [n] [model]
  which in {quick, fresh, holdout}  (default: quick)

Reads results/agent_<model>_<which>_<n>.json + its trace, finds examples whose
trace contains an `llm_error` event (or whose prediction is empty with an error
source), re-runs ONLY those with the current (hardened-retry) client, updates the
rows + metrics in place, and rewrites the JSON. A timestamped backup is kept.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, evaluator
from src.agent import load_prompts, run_example
from src.config import load_llm_config, load_verifier_config
from src.evaluator import check_denotation, to_value_list
from src.llm import LLMClient
from src.schemas import Budget
from src.tools.wtq_tools import build_registry
from src.trace import Tracer


def _is_correct(gold, pred) -> bool:
    try:
        return check_denotation(to_value_list(gold), to_value_list(pred))
    except Exception:  # noqa: BLE001
        return False

SPLIT = "random-split-1-dev"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _error_ids_from_trace(trace_path: str) -> set[str]:
    """Example ids whose trace contains an llm_error event."""
    ids: set[str] = set()
    if not os.path.exists(trace_path):
        return ids
    with open(trace_path, encoding="utf8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(e.get("kind") == "llm_error" for e in r.get("events", [])):
                ids.add(r.get("example_id"))
    ids.discard(None)
    return ids


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "quick"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    model = sys.argv[3] if len(sys.argv) > 3 else "qwen3.6-35b-a3b"

    res_path = os.path.join(RESULTS_DIR, f"agent_{model}_{which}_{n}.json")
    trace_path = os.path.join(RESULTS_DIR, f"trace_agent_{model}_{which}_{n}.jsonl")
    if not os.path.exists(res_path):
        print(f"results file not found: {res_path}")
        return 2

    with open(res_path, encoding="utf8") as f:
        result_doc = json.load(f)
    rows = result_doc["rows"]
    row_by_id = {r["id"]: r for r in rows}

    err_ids = _error_ids_from_trace(trace_path)
    # also catch empty predictions whose source signals an error/fallback
    for r in rows:
        if not r.get("pred") and (r.get("src") in (None, "empty_fallback") or
                                  (r.get("verify") is None and r.get("steps") in (None, 0))):
            err_ids.add(r["id"])
    # Only re-run errored examples that are currently WRONG — re-running an
    # already-correct one risks flipping it (model nondeterminism).
    err_ids = {i for i in err_ids if i in row_by_id
               and not _is_correct(row_by_id[i]["gold"], row_by_id[i]["pred"])}
    if not err_ids:
        print("no wrong+errored examples found — nothing to re-run.")
        return 0
    print(f"re-running {len(err_ids)} errored+wrong example(s): {sorted(err_ids)}")

    cfg = load_llm_config(model=model)
    client = LLMClient(cfg)
    vcfg = load_verifier_config()
    verifier_client = LLMClient(vcfg)
    prompts = load_prompts()
    registry = build_registry()
    budget = Budget(max_steps=20)

    examples = {e.id: e for e in data.eval_subset(data.load_examples(SPLIT), n, which)}
    tagged = evaluator.find_tagged_path(data.DEFAULT_DATASET_ROOT, SPLIT)
    targets_all = evaluator.load_targets_from_tagged(tagged)

    shutil.copy(res_path, res_path.replace(".json", f".pre_rerun_{int(time.time())}.json"))

    t0 = time.time()
    fixed = 0
    for k, eid in enumerate(sorted(err_ids), 1):
        ex = examples.get(eid)
        if ex is None:
            print(f"  [{k}/{len(err_ids)}] {eid}: not in subset, skip")
            continue
        ex_t0 = time.time()
        try:
            tc = data.load_table(ex.table_path)
            tracer = Tracer(trace_path + ".rerun", ex.id)
            pred = run_example(ex, tc, registry, client, prompts, budget=budget,
                               tracer=tracer, verifier_client=verifier_client)
            items = pred.items
            ev = pred.evidence
        except Exception as exc:  # noqa: BLE001
            items, ev = [], {"error": f"{type(exc).__name__}: {exc}"}
        r = row_by_id[eid]
        old = r.get("pred")
        r["pred"] = items
        r["src"] = ev.get("answer_source")
        r["steps"] = ev.get("steps_used")
        r["verify"] = ev.get("verify")
        r["verify_retries"] = ev.get("verify_retries")
        r["candidates"] = ev.get("candidates")
        if items:
            fixed += 1
        print(f"  [{k}/{len(err_ids)}] {eid}: {old} -> {items} gold={ex.target_value} "
              f"({time.time()-ex_t0:.1f}s)", flush=True)

    predictions = {r["id"]: r["pred"] for r in rows}
    targets = {eid: targets_all[eid] for eid in examples if eid in targets_all}
    disputed = data.load_disputed()
    new_metrics = evaluator.evaluate(predictions, targets, exclude_ids=set(disputed))
    result_doc["metrics"] = {k: v for k, v in new_metrics.items() if k != "per_example"}

    with open(res_path, "w", encoding="utf8") as f:
        json.dump(result_doc, f, ensure_ascii=False, indent=2)

    print(f"\nre-ran {len(err_ids)}, recovered non-empty for {fixed} | {time.time()-t0:.0f}s")
    print("==== UPDATED RESULT ====")
    print(f"accuracy (raw)      = {new_metrics['accuracy']}  "
          f"({new_metrics['num_correct']}/{new_metrics['num_examples']})")
    print(f"accuracy (adjusted) = {new_metrics['accuracy_adjusted']}  "
          f"({new_metrics['num_correct_adjusted']}/{new_metrics['num_examples_adjusted']}, "
          f"excluded {new_metrics['num_excluded_disputed']} disputed)")
    print(f"saved = {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
