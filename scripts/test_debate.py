"""test_debate.py — targeted multi-round debate tester.

Runs a focused set of examples where the two holdout runs disagreed and prints
the FULL discussion trace: every step, each debate round, auditor's response.

Usage:
  python scripts/test_debate.py [--ids nt-5881 nt-7770 ...] [--max-rounds 2]

Default target list:
  - Regressions (run_b worse than run_a): nt-5881 nt-7770 nt-2445 nt-7226 nt-7596 nt-1239
  - Improvements (run_b better):         nt-12721 nt-13292 nt-13784 nt-5569
  - Both-wrong (different answers):      nt-11294 nt-962 nt-13819
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.config import load_llm_config, load_verifier_config
from src.data import load_examples, load_table
from src.llm import LLMClient
from src.schemas import Budget
from src.agent import load_prompts, run_example
from src.evaluator import evaluate, check_denotation, to_value_list
from src.tools.wtq_tools import build_registry

# --------------------------------------------------------------------------
# Example categories (from comparison of agent_holdout_200_2b vs _holdout_200)
# --------------------------------------------------------------------------
REGRESSIONS = ["nt-5881", "nt-7770", "nt-2445", "nt-7226", "nt-7596", "nt-1239"]
IMPROVEMENTS = ["nt-12721", "nt-13292", "nt-13784", "nt-5569"]
BOTH_WRONG   = ["nt-11294", "nt-962", "nt-13819"]

DEFAULT_IDS = REGRESSIONS + IMPROVEMENTS + BOTH_WRONG

# Previous answers for context display
PREV_ANSWERS: dict[str, dict] = {
    "nt-5881":  {"run_a": ["1512"], "run_b": ["1588"], "gold": ["1512"]},
    "nt-7770":  {"run_a": ["August 1"], "run_b": ["August 9"], "gold": ["August 1"]},
    "nt-2445":  {"run_a": ["I.N.F.O. & NOVA"], "run_b": ["None"], "gold": ["I.N.F.O. & NOVA"]},
    "nt-7226":  {"run_a": ["7"], "run_b": ["4"], "gold": ["7"]},
    "nt-7596":  {"run_a": ["Tampa Bay Lightning"], "run_b": ["Hartford Whalers"], "gold": ["Tampa Bay Lightning"]},
    "nt-1239":  {"run_a": ["1926"], "run_b": ["1925"], "gold": ["1926"]},
    "nt-12721": {"run_a": ["at Denver Broncos"], "run_b": ["Denver Broncos"], "gold": ["Denver Broncos"]},
    "nt-13292": {"run_a": ["11"], "run_b": ["10"], "gold": ["10"]},
    "nt-13784": {"run_a": ["5 acts"], "run_b": ["5"], "gold": ["5"]},
    "nt-5569":  {"run_a": ["7"], "run_b": ["8"], "gold": ["8"]},
    "nt-11294": {"run_a": ["12"], "run_b": ["10"], "gold": ["11"]},
    "nt-962":   {"run_a": ["12"], "run_b": ["14"], "gold": ["13"]},
    "nt-13819": {"run_a": ["0"], "run_b": ["2500"], "gold": ["2500"]},
}


def _wrong_ids_from_results(path: str) -> list[str]:
    """Extract IDs that a prior run got wrong (using official targets, excluding disputed)."""
    from src import data as _data, evaluator as _ev
    with open(path) as f:
        d = json.load(f)
    rows = d.get("rows", d) if isinstance(d, dict) else d
    tagged = _ev.find_tagged_path(_data.DEFAULT_DATASET_ROOT, "random-split-1-dev")
    targets = _ev.load_targets_from_tagged(tagged)
    disputed = set(_data.load_disputed())
    wrong = []
    for r in rows:
        eid = r["id"]
        if eid in disputed:
            continue
        tgt = targets.get(eid)
        pred = to_value_list(r.get("pred") or [])
        ok = check_denotation(tgt, pred) if tgt else check_denotation(to_value_list(r.get("gold") or []), pred)
        if not ok:
            wrong.append(eid)
    return wrong


def divider(ch: str = "─", width: int = 72) -> str:
    return ch * width


def banner(text: str, ch: str = "═") -> str:
    return f"\n{ch * 72}\n  {text}\n{ch * 72}"


def print_trace(trace_events: list[dict]) -> None:
    for ev in trace_events:
        if isinstance(ev, dict):
            kind = ev.get("kind", "?")
            step = ev.get("step", "?")
            note = ev.get("note", "")
            tc = ev.get("tool_call")
            # tc may be a ToolCall dataclass or a dict
            if tc is not None:
                if isinstance(tc, dict):
                    tc_name = tc.get("name", "?")
                    tc_args = tc.get("args", {})
                else:
                    tc_name = getattr(tc, "name", "?")
                    tc_args = getattr(tc, "args", {})
            else:
                tc_name, tc_args = None, {}
        else:
            # TraceEvent dataclass
            kind = getattr(ev, "kind", "?")
            step = getattr(ev, "step", "?")
            note = getattr(ev, "note", "")
            tc_raw = getattr(ev, "tool_call", None)
            if tc_raw is not None:
                tc_name = getattr(tc_raw, "name", "?")
                tc_args = getattr(tc_raw, "args", {})
            else:
                tc_name, tc_args = None, {}

        if kind == "llm_response":
            print(f"  [step {step}] LLM → {str(note)[:120]}")
        elif kind == "tool_call" and tc_name:
            if tc_name == "submit_answer":
                print(f"  [step {step}] SUBMIT → {tc_args.get('items', []) if isinstance(tc_args, dict) else getattr(tc_args, 'items', [])}")
            elif tc_name == "run_python":
                code = (tc_args.get("code", "") if isinstance(tc_args, dict) else getattr(tc_args, "code", ""))
                print(f"  [step {step}] run_python → {str(code)[:120].replace(chr(10), ' ')}…")
            else:
                print(f"  [step {step}] {tc_name}")
        elif kind == "verify":
            print(f"  [step {step}] VERIFY → {note}")
        elif kind == "debate_start":
            print(f"  [step {step}] DEBATE START → {note}")
        elif kind == "llm_error":
            print(f"  [step {step}] LLM ERROR → {note}")


def run_single(ex, client: LLMClient, verifier_client: LLMClient,
               registry, prompts, budget: Budget) -> dict:
    from src.data import load_table
    from src.trace import Tracer

    table_ctx = load_table(ex.table_path)
    trace_path = str(ROOT / "results" / f"trace_debate_{ex.id}.jsonl")
    tracer = Tracer(out_path=trace_path, example_id=ex.id)

    pred = run_example(ex, table_ctx, registry, client, prompts,
                       budget=budget, tracer=tracer, verifier_client=verifier_client)

    correct = check_denotation(to_value_list(ex.target_value), to_value_list(pred.items))
    ev = pred.evidence
    return {
        "id": ex.id,
        "q": ex.utterance,
        "pred": pred.items,
        "gold": ex.target_value,
        "correct": correct,
        "src": ev.get("answer_source"),
        "verify": ev.get("verify"),
        "steps": ev.get("steps_used"),
        "trace": [e.__dict__ for e in tracer.events],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", default=None,
                        help="Example IDs to test (default: all 13 target examples)")
    parser.add_argument("--from-wrong", type=str, default=None,
                        help="Path to a results JSON; re-run all examples that run got WRONG "
                             "(excludes disputed). Overrides --ids.")
    parser.add_argument("--max-rounds", type=int, default=2,
                        help="max_verify_retries (debate rounds)")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Agent step budget")
    parser.add_argument("--debate-steps", type=int, default=3,
                        help="Extra steps per debate round")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    if args.from_wrong:
        target_ids = _wrong_ids_from_results(args.from_wrong)
        print(f"[from-wrong] {args.from_wrong}: {len(target_ids)} wrong examples to re-test")
    else:
        target_ids = args.ids or DEFAULT_IDS
    budget = Budget(
        max_steps=args.max_steps,
        max_verify_retries=args.max_rounds,
        debate_extra_steps=args.debate_steps,
    )

    cfg = load_llm_config()
    vcfg = load_verifier_config()
    client = LLMClient(cfg)
    verifier_client = LLMClient(vcfg)
    registry = build_registry()
    prompts = load_prompts()

    print(banner(f"MULTI-ROUND DEBATE TEST  (max_rounds={args.max_rounds}, max_steps={args.max_steps})"))
    print(f"Target IDs ({len(target_ids)}): {' '.join(target_ids)}")
    print(f"Generator model : {cfg.model}")
    print(f"Verifier  model : {vcfg.model}")
    print()

    # Load all examples from the holdout split
    all_examples = load_examples("random-split-1-dev")
    example_map = {ex.id: ex for ex in all_examples}

    missing = [eid for eid in target_ids if eid not in example_map]
    if missing:
        print(f"WARNING: IDs not found in dev split: {missing}")
        target_ids = [eid for eid in target_ids if eid in example_map]

    results = []
    n_correct = 0

    for i, eid in enumerate(target_ids, 1):
        ex = example_map[eid]
        prev = PREV_ANSWERS.get(eid, {})

        print(divider("─"))
        print(f"[{i}/{len(target_ids)}] {eid}")
        print(f"  Question : {ex.utterance}")
        print(f"  Gold     : {ex.target_value}")
        if prev:
            print(f"  run_a    : {prev.get('run_a', '?')}  ({'✓' if prev.get('run_a') == prev.get('gold') else '✗'})")
            print(f"  run_b    : {prev.get('run_b', '?')}  ({'✓' if prev.get('run_b') == prev.get('gold') else '✗'})")
        print()

        try:
            result = run_single(ex, client, verifier_client, registry, prompts, budget)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            results.append({"id": eid, "error": str(exc)})
            continue

        correct = result["correct"]
        n_correct += correct
        mark = "✓" if correct else "✗"
        print(f"\n  NEW ANSWER: {result['pred']}  [{mark}]  (src={result['src']})")
        if result.get("verify"):
            v = result["verify"]
            print(f"  VERIFY: ok={v.get('ok')} src={v.get('source')} issues={v.get('issues', [])}")

        print("\n  — Full trace —")
        print_trace(result.get("trace", []))

        results.append(result)
        print()

    print(divider("═"))
    print(f"\nSUMMARY: {n_correct}/{len(results)} correct  "
          f"({100*n_correct/max(len(results),1):.1f}%)")
    print()

    # Per-example breakdown
    for r in results:
        if "error" in r:
            print(f"  {r['id']}  ERROR")
            continue
        mark = "✓" if r["correct"] else "✗"
        prev = PREV_ANSWERS.get(r["id"], {})
        a_ok = prev.get("run_a") == prev.get("gold")
        b_ok = prev.get("run_b") == prev.get("gold")
        verdict = ""
        if a_ok and not b_ok and r["correct"]:
            verdict = "← REGRESSION FIXED"
        elif a_ok and not b_ok and not r["correct"]:
            verdict = "← REGRESSION PERSISTS"
        elif not a_ok and b_ok and r["correct"]:
            verdict = "(improvement maintained)"
        elif not a_ok and b_ok and not r["correct"]:
            verdict = "(improvement lost)"
        print(f"  {r['id']}  {mark}  pred={r['pred']}  gold={r['gold']}  {verdict}")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"budget": budget.__dict__, "results": results}, f,
                      ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
