"""Phase 0 smoke test (no LLM required).

Verifies the no-model foundation end to end:
  1. Load a split + sample the quick set.
  2. Load every referenced table as a DataFrame.
  3. Build authoritative targets from the .tagged file.
  4. Evaluator alignment check: feeding the gold answers back as predictions
     must score 100% (proves our matching logic matches the official one).
  5. A trivial sandbox round-trip.

Run:  python -m scripts.smoke
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, evaluator
from src.sandbox import run_code

SPLIT = "random-split-1-dev"
QUICK_N = 200


def main() -> int:
    root = data.DEFAULT_DATASET_ROOT
    print(f"[1] Loading split '{SPLIT}' from {root} ...")
    examples = data.load_examples(SPLIT)
    print(f"    loaded {len(examples)} examples")

    quick = data.sample_examples(examples, QUICK_N)
    print(f"[1b] Quick set: {len(quick)} examples (seed-deterministic)")

    print("[2] Loading all referenced tables in the quick set ...")
    tables_ok, tables_fail = 0, []
    for ex in quick:
        try:
            tc = data.load_table(ex.table_path)
            assert tc.n_rows >= 0 and tc.columns
            tables_ok += 1
        except Exception as exc:  # noqa: BLE001
            tables_fail.append((ex.table_path, str(exc)))
    print(f"    tables loaded OK: {tables_ok}, failed: {len(tables_fail)}")
    for path, err in tables_fail[:5]:
        print(f"      FAIL {path}: {err}")

    print("[3] Building authoritative targets from .tagged ...")
    tagged_path = evaluator.find_tagged_path(root, SPLIT)
    if not tagged_path:
        print("    ERROR: tagged file not found")
        return 1
    targets_all = evaluator.load_targets_from_tagged(tagged_path)
    targets = {ex.id: targets_all[ex.id] for ex in quick if ex.id in targets_all}
    print(f"    targets for quick set: {len(targets)} / {len(quick)}")

    print("[4] Evaluator alignment: gold-as-prediction must be ~100% ...")
    gold_predictions = {ex.id: ex.target_value for ex in quick if ex.id in targets}
    result = evaluator.evaluate(gold_predictions, targets)
    print(f"    accuracy={result['accuracy']} "
          f"({result['num_correct']}/{result['num_examples']})")

    # An empty/garbage prediction must NOT be correct (sanity in the other direction).
    bad = {ex_id: ["__definitely_wrong__"] for ex_id in targets}
    bad_result = evaluator.evaluate(bad, targets)
    print(f"    sanity (garbage preds) accuracy={bad_result['accuracy']}")

    print("[5] Sandbox round-trip ...")
    tc = data.load_table(quick[0].table_path)
    exec_res = run_code("answer = [str(len(df))]\nevidence = {'n_rows': len(df)}", tc.df)
    print(f"    sandbox ok={exec_res.ok} answer={exec_res.answer} err={exec_res.error}")

    passed = (
        len(tables_fail) == 0
        and result["accuracy"] >= 0.99
        and bad_result["accuracy"] <= 0.05
        and exec_res.ok
    )
    print("\nSMOKE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
