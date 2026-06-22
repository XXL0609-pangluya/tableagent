"""Run the direct-answer baseline on an eval subset and report accuracy.

Run:  python -m scripts.run_baseline [n] [which] [model]
  which in {quick, fresh, holdout}
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, evaluator
from src.baselines import run_direct_answer
from src.config import load_llm_config
from src.llm import LLMClient

SPLIT = "random-split-1-dev"
QUICK_N = 200
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    which = sys.argv[2] if len(sys.argv) > 2 else "quick"
    model = sys.argv[3] if len(sys.argv) > 3 else None

    cfg = load_llm_config(model=model)
    client = LLMClient(cfg)
    print(f"model={cfg.model}  n={n}  set={which}  split={SPLIT}")

    examples = data.eval_subset(data.load_examples(SPLIT), n, which)
    tagged = evaluator.find_tagged_path(data.DEFAULT_DATASET_ROOT, SPLIT)
    targets_all = evaluator.load_targets_from_tagged(tagged)

    predictions: dict[str, list[str]] = {}
    total_tokens = 0
    rows = []
    t0 = time.time()
    for i, ex in enumerate(examples, 1):
        try:
            tc = data.load_table(ex.table_path)
            pred, usage = run_direct_answer(ex, tc, client)
        except Exception as exc:
            from src.schemas import Prediction
            pred, usage = Prediction(id=ex.id, items=[], evidence={"error": str(exc)}), {}
        predictions[ex.id] = pred.items
        total_tokens += usage.get("total_tokens", 0)
        rows.append({
            "id": ex.id, "q": ex.utterance,
            "pred": pred.items, "gold": ex.target_value,
        })
        print(f"  [{i}/{n}] {ex.id}: pred={pred.items} gold={ex.target_value}", flush=True)

    targets = {ex.id: targets_all[ex.id] for ex in examples if ex.id in targets_all}
    disputed = data.load_disputed()
    result = evaluator.evaluate(predictions, targets, exclude_ids=set(disputed))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"baseline_direct_{cfg.model}_{which}_{n}.json")
    with open(out, "w", encoding="utf8") as f:
        json.dump({"config": {"model": cfg.model, "n": n, "set": which, "split": SPLIT},
                   "metrics": {k: v for k, v in result.items() if k != "per_example"},
                   "total_tokens": total_tokens,
                   "elapsed_s": round(time.time() - t0, 1),
                   "rows": rows}, f, ensure_ascii=False, indent=2)

    print("\n==== RESULT ====")
    print(f"accuracy (raw)      = {result['accuracy']}  ({result['num_correct']}/{result['num_examples']})")
    print(f"accuracy (adjusted) = {result['accuracy_adjusted']}  "
          f"({result['num_correct_adjusted']}/{result['num_examples_adjusted']}, "
          f"excluded {result['num_excluded_disputed']} disputed)")
    print(f"total_tokens = {total_tokens}")
    print(f"elapsed      = {round(time.time() - t0, 1)}s")
    print(f"saved        = {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
