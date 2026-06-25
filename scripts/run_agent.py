"""Run the function-calling agent (Phase 2b: skills + verifier) on an eval subset.

Run:  python -m scripts.run_agent [n] [which] [model]
  which in {quick, fresh, holdout}
    holdout = third disjoint 200 (excludes quick+fresh) — use for new test runs
Defaults to n=20. Writes predictions + per-example traces to results/.

Checkpoint/resume: progress is saved to a .partial.json after EVERY example.
If the run is interrupted (crash, Ctrl-C, network drop), just re-run the SAME
command and it will skip already-finished examples and continue. The checkpoint
is removed automatically once the run completes.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, evaluator
from src.agent import load_prompts, run_example
from src.config import load_llm_config, load_verifier_config
from src.llm import LLMClient
from src.schemas import Budget
from src.tools.wtq_tools import build_registry
from src.trace import Tracer

SPLIT = "random-split-1-dev"
QUICK_N = 200
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _save_checkpoint(path: str, rows: list, t0: float) -> None:
    """Atomically write progress so an interrupted run can resume."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump({"elapsed_s": round(time.time() - t0, 1), "rows": rows},
                  f, ensure_ascii=False)
    os.replace(tmp, path)


def _load_checkpoint(path: str) -> tuple[list, dict, float]:
    """Return (rows, predictions_by_id, prior_elapsed_s) from an existing checkpoint."""
    if not os.path.exists(path):
        return [], {}, 0.0
    try:
        with open(path, "r", encoding="utf8") as f:
            data_ck = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], {}, 0.0
    rows = data_ck.get("rows", [])
    preds = {r["id"]: r["pred"] for r in rows if "id" in r}
    return rows, preds, float(data_ck.get("elapsed_s", 0.0))


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    which = sys.argv[2] if len(sys.argv) > 2 else "quick"
    model = sys.argv[3] if len(sys.argv) > 3 else None

    cfg = load_llm_config(model=model)
    client = LLMClient(cfg)
    vcfg = load_verifier_config()
    verifier_client = LLMClient(vcfg)
    prompts = load_prompts()
    registry = build_registry()
    budget = Budget(max_steps=20)
    print(f"model={cfg.model}  verifier={vcfg.model}  n={n}  set={which}  split={SPLIT}  "
          f"max_steps={budget.max_steps}  verify_retries={budget.max_verify_retries}")

    examples = data.eval_subset(data.load_examples(SPLIT), n, which)
    tagged = evaluator.find_tagged_path(data.DEFAULT_DATASET_ROOT, SPLIT)
    targets_all = evaluator.load_targets_from_tagged(tagged)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    trace_path = os.path.join(RESULTS_DIR, f"trace_agent_{cfg.model}_{which}_{n}.jsonl")
    ckpt_path = os.path.join(RESULTS_DIR, f"agent_{cfg.model}_{which}_{n}.partial.json")

    # Resume from checkpoint if present.
    rows, predictions, prior_elapsed = _load_checkpoint(ckpt_path)
    done_ids = set(predictions)
    if done_ids:
        print(f"[resume] found checkpoint with {len(done_ids)} finished examples — "
              f"skipping them and continuing.", flush=True)
    elif os.path.exists(trace_path):
        os.remove(trace_path)  # fresh run: clear stale traces

    t0 = time.time()  # wall clock for THIS process
    total = len(examples)
    for i, ex in enumerate(examples, 1):
        if ex.id in done_ids:
            continue
        ex_t0 = time.time()
        tracer = None
        try:
            tc = data.load_table(ex.table_path)
            tracer = Tracer(trace_path, ex.id)
            pred = run_example(ex, tc, registry, client, prompts, budget=budget, tracer=tracer,
                               verifier_client=verifier_client)
        except Exception as exc:  # keep a long run alive on a single bad example
            from src.schemas import Prediction
            pred = Prediction(id=ex.id, items=[], evidence={"error": f"{type(exc).__name__}: {exc}"})
        # Trace logging is isolated: a flush failure must NEVER discard a good answer.
        if tracer is not None:
            tracer.flush(extra={"question": ex.utterance, "gold": ex.target_value, "pred": pred.items})
        predictions[ex.id] = pred.items
        rows.append({"id": ex.id, "q": ex.utterance, "pred": pred.items,
                     "gold": ex.target_value, "src": pred.evidence.get("answer_source"),
                     "steps": pred.evidence.get("steps_used"),
                     "verify": pred.evidence.get("verify"),
                     "verify_retries": pred.evidence.get("verify_retries"),
                     "candidates": pred.evidence.get("candidates"),
                     "skills": pred.evidence.get("skills"),
                     "error": pred.evidence.get("error")})
        _save_checkpoint(ckpt_path, rows, t0)

        vnote = " verify=FAIL" if (pred.evidence.get("verify") and not pred.evidence["verify"].get("ok")) else ""
        # Timing + ETA based on examples processed in THIS session.
        ex_dt = time.time() - ex_t0
        done_this_session = len([r for r in rows if r["id"] not in done_ids])
        avg = (time.time() - t0) / max(1, done_this_session)
        remaining = total - i
        eta = _fmt_hms(remaining * avg)
        print(f"  [{i}/{total}] {ex.id}: pred={pred.items} gold={ex.target_value} "
              f"(src={pred.evidence.get('answer_source')}, steps={pred.evidence.get('steps_used')}){vnote} "
              f"| {ex_dt:.1f}s avg={avg:.1f}s ETA={eta}",
              flush=True)

    targets = {ex.id: targets_all[ex.id] for ex in examples if ex.id in targets_all}
    disputed = data.load_disputed()
    result = evaluator.evaluate(predictions, targets, exclude_ids=set(disputed))

    out = os.path.join(RESULTS_DIR, f"agent_{cfg.model}_{which}_{n}.json")
    with open(out, "w", encoding="utf8") as f:
        json.dump({"config": {"model": cfg.model, "verifier_model": vcfg.model,
                              "n": n, "set": which, "split": SPLIT,
                              "max_steps": budget.max_steps, "phase": "2c",
                              "max_verify_retries": budget.max_verify_retries},
                   "metrics": {k: v for k, v in result.items() if k != "per_example"},
                   "elapsed_s": round(time.time() - t0, 1),
                   "rows": rows}, f, ensure_ascii=False, indent=2)

    # Run finished cleanly — remove the checkpoint.
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print("\n==== RESULT ====")
    print(f"accuracy (raw)      = {result['accuracy']}  ({result['num_correct']}/{result['num_examples']})")
    print(f"accuracy (adjusted) = {result['accuracy_adjusted']}  "
          f"({result['num_correct_adjusted']}/{result['num_examples_adjusted']}, "
          f"excluded {result['num_excluded_disputed']} disputed)")
    print(f"elapsed  = {_fmt_hms(time.time() - t0)} (this session)")
    print(f"saved    = {out}")
    print(f"traces   = {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
