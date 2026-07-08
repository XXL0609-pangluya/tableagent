"""run_hitab_pilot.py — HiTab pilot runner (HiTab-specific).

Reuses the dataset-agnostic core (`run_example` = generator loop + verifier debate +
finalize) with HiTab-specific loading (`src/hitab_data.py`) and matching
(`src/hitab_eval.py`). WTQ code paths are untouched.

Usage:
  python scripts/run_hitab_pilot.py --split dev --n 20 --save results/hitab_pilot_dev20.json
  python scripts/run_hitab_pilot.py --n 100 --max-steps 20 --max-rounds 2

Checkpoint/resume: partial results are written to <save>.partial.json after each
example so a long run can be resumed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.config import load_llm_config, load_verifier_config
from src.llm import LLMClient
from src.schemas import Budget
from src.agent import load_prompts, run_example
from src.tools.wtq_tools import build_registry  # dataset-agnostic: operates on any df
from src.hitab_data import load_hitab_examples, load_hitab_table
from src.hitab_eval import answer_match
from src.trace import Tracer

_HITAB_SKILL_PATH = ROOT / "prompts" / "skills" / "hitab.md"
_HITAB_MODULE_DIR = ROOT / "prompts" / "skills" / "hitab"

# Progressive-disclosure routing: an always-on `core` module (prepended to the
# charter) plus per-question modules injected only when the question matches.
# Keeps each question's HiTab guidance short AND relevant instead of dumping the
# whole convention list every time. Auditable keyword triggers, mirroring
# agent._SKILL_TRIGGERS.
_HITAB_MODULE_TRIGGERS: dict[str, tuple[str, ...]] = {
    "percent": (r"\bpercent", r"\bpercentage", r"\bproportion", r"\bshare\b", r"per cent"),
    "sign": (r"\bchang", r"\bdecreas", r"\bdeclin", r"\bdrop", r"\bfell\b", r"\bfall",
             r"\bgap\b", r"\bdifference", r"\bincreas", r"\brose\b", r"\bgrew\b",
             r"\bgrowth\b", r"less likely", r"more likely", r"\bfewer\b",
             r"\breduction\b", r"\brise\b"),
    "label": (r"which group", r"which region", r"which sector", r"\bwhich\b",
              r"\bwho\b", r"what group", r"\bwhose\b"),
    "format": (r"\brecord\b", r"respectively", r"won-?lost", r"win-?loss",
               r"\blist\b", r"combined", r"together", r"\bboth\b", r"\btop \b"),
    "ratio": (r"\btimes\b",),
    "units": (r"\bmillion", r"\bbillion", r"\bthousand", r"\bdollar", r"per 100",
              r"\brate\b", r"\$"),
    "rows": (r"\bclub\b", r"\bteam\b", r"play(ed)? for", r"\btop\b", r"\bhighest\b",
             r"\blowest\b", r"\brank", r"\bmost\b", r"\bleast\b", r"\blargest\b",
             r"\bsmallest\b"),
}


def load_hitab_prompts() -> "Prompts":
    """Load the shared prompts and attach HiTab-specific guidance. WTQ code paths
    are untouched (they call `load_prompts` directly).

    Modes (env HITAB_SKILL_MODE, default 'mono'):
      - 'routed': prepend prompts/skills/hitab/core.md to the charter and register
        the per-question modules (progressive disclosure). Falls back to mono if
        the hitab/ module dir is missing.
      - 'mono':   prepend the whole prompts/skills/hitab.md (legacy behaviour).
    HITAB_SKILL_FILE (absolute or name under prompts/skills/) forces mono with a
    specific file — useful for A/B testing a variant WITHOUT overwriting anything.
    """
    prompts = load_prompts()
    override = os.environ.get("HITAB_SKILL_FILE", "").strip()
    mode = os.environ.get("HITAB_SKILL_MODE", "mono").strip().lower()

    if not override and mode == "routed" and (_HITAB_MODULE_DIR / "core.md").exists():
        core = (_HITAB_MODULE_DIR / "core.md").read_text(encoding="utf8")
        prompts.charter = core + "\n\n---\n\n" + prompts.charter
        modules: list[tuple[str, tuple[str, ...], str]] = []
        for name, patterns in _HITAB_MODULE_TRIGGERS.items():
            p = _HITAB_MODULE_DIR / f"{name}.md"
            if p.exists():
                modules.append((f"hitab:{name}", patterns, p.read_text(encoding="utf8")))
        prompts.dynamic_modules = modules
        print(f"[hitab-skill] routed: core ({core.count(chr(10))+1} lines) + "
              f"{len(modules)} on-demand modules {[m[0] for m in modules]}")
        return prompts

    # Mono mode (legacy) or explicit file override.
    skill_path = _HITAB_SKILL_PATH
    if override:
        skill_path = Path(override)
        if not skill_path.is_absolute():
            skill_path = ROOT / "prompts" / "skills" / override
    if skill_path.exists():
        hitab_skill = skill_path.read_text(encoding="utf8")
        prompts.charter = hitab_skill + "\n\n---\n\n" + prompts.charter
        print(f"[hitab-skill] mono: using {skill_path} ({hitab_skill.count(chr(10))+1} lines)")
    else:
        print(f"[hitab-skill] WARNING: skill file not found: {skill_path}")
    return prompts


def _ckpt_path(save_path: str | Path) -> Path:
    p = Path(save_path)
    return p.with_suffix(p.suffix + ".partial.json")


def _load_ckpt(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    rows = doc.get("results", [])
    return rows if isinstance(rows, list) else []


def _save(path: Path, results: list[dict], budget: Budget, gen_model: str, ver_model: str,
          split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n_correct = sum(1 for r in results if r.get("correct"))
    payload = {
        "dataset": "HiTab",
        "split": split,
        "config": {"generator_model": gen_model, "verifier_model": ver_model},
        "budget": budget.__dict__,
        "metrics": {"n": len(results), "correct": n_correct,
                    "accuracy": n_correct / max(1, len(results))},
        "results": results,
    }
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def preflight(name: str, client: LLMClient) -> bool:
    try:
        resp = client.chat(messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                           max_tokens=16, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        print(f"{name} preflight FAILED: {type(exc).__name__}: {exc}")
        return False
    print(f"{name} preflight OK (reply={(resp.text or '').strip()!r})")
    return True


def run_single(ex, client, verifier_client, registry, prompts, budget) -> dict:
    table_ctx = load_hitab_table(ex.table_path)
    # Trace path is configurable via env so each iteration can write to its own
    # directory/filename WITHOUT overwriting previous iterations' traces.
    #   HITAB_TRACE_DIR  -> directory (default: results/HiTab)
    #   HITAB_TRACE_TAG  -> filename infix (default: none) -> trace_hitab[_<tag>]_<id>.jsonl
    _trace_dir = os.environ.get("HITAB_TRACE_DIR", str(ROOT / "results" / "HiTab"))
    _trace_tag = os.environ.get("HITAB_TRACE_TAG", "").strip()
    _infix = f"_{_trace_tag}" if _trace_tag else ""
    trace_path = str(Path(_trace_dir) / f"trace_hitab{_infix}_{ex.id}.jsonl")
    os.makedirs(_trace_dir, exist_ok=True)
    tracer = Tracer(out_path=trace_path, example_id=ex.id)
    pred = run_example(ex, table_ctx, registry, client, prompts,
                       budget=budget, tracer=tracer, verifier_client=verifier_client)
    # ALWAYS flush the tracer — without this the in-memory events are discarded
    # and no trace file is ever written (this was the round-1 bug: 200 runs left
    # zero trace files). Flush with meta so the trace is self-describing.
    if tracer is not None:
        tracer.flush(extra={"question": ex.utterance, "gold": ex.target_value,
                            "pred": pred.items, "table_id": ex.table_path})
    correct = answer_match(pred.items, ex.target_value)
    ev = pred.evidence
    return {
        "id": ex.id,
        "q": ex.utterance,
        "table_id": ex.table_path,
        "pred": pred.items,
        "gold": ex.target_value,
        "correct": correct,
        "src": ev.get("answer_source"),
        "verify": ev.get("verify"),
        "verify_retries": ev.get("verify_retries"),
        "candidates": ev.get("candidates"),
        "steps": ev.get("steps_used"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n", type=int, default=20, help="number of examples (from the front of the split)")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--max-rounds", type=int, default=2, help="max_verify_retries (debate rounds)")
    ap.add_argument("--debate-steps", type=int, default=3)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--ids-file", type=str, default=None,
                    help="JSON list of example ids to run (filtered from the split). "
                         "Used to re-run only the failed subset between iterations.")
    args = ap.parse_args()

    # Enable HiTab answer-convention checks in the verifier (gated off for WTQ).
    os.environ.setdefault("HITAB_VERIFY", "1")

    budget = Budget(max_steps=args.max_steps, max_verify_retries=args.max_rounds,
                    debate_extra_steps=args.debate_steps)

    cfg = load_llm_config()
    vcfg = load_verifier_config()
    client = LLMClient(cfg)
    verifier_client = LLMClient(vcfg)
    registry = build_registry()
    prompts = load_hitab_prompts()

    print(f"=== HiTab pilot: split={args.split} n={args.n} "
          f"max_steps={args.max_steps} max_rounds={args.max_rounds} ===")
    print(f"Generator: {cfg.model}   Verifier: {vcfg.model}")
    if not (preflight("solver", client) and preflight("verifier", verifier_client)):
        raise SystemExit(2)

    examples = load_hitab_examples(args.split)
    if args.ids_file:
        with open(args.ids_file, encoding="utf8") as f:
            wanted = set(json.load(f))
        examples = [ex for ex in examples if ex.id in wanted]
        print(f"[ids-file] filtering to {len(wanted)} ids -> {len(examples)} matched in split")
    else:
        examples = examples[: args.n]
    print(f"Loaded {len(examples)} examples\n")

    results: list[dict] = []
    done: set[str] = set()
    if args.save:
        results = _load_ckpt(_ckpt_path(args.save))
        done = {r["id"] for r in results if r.get("id")}
        if done:
            print(f"[resume] {len(done)} already done — skipping\n")

    for i, ex in enumerate(examples, 1):
        if ex.id in done:
            continue
        try:
            r = run_single(ex, client, verifier_client, registry, prompts, budget)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(examples)}] {ex.id} ERROR: {exc}", flush=True)
            r = {"id": ex.id, "error": str(exc)}
        results.append(r)
        if args.save:
            _save(_ckpt_path(args.save), results, budget, cfg.model, vcfg.model, args.split)
        mark = "OK " if r.get("correct") else ("ERR" if "error" in r else "  X")
        print(f"[{i}/{len(examples)}] {mark} {ex.id}  pred={r.get('pred')}  gold={r.get('gold')}  "
              f"(src={r.get('src')}, rounds={r.get('verify_retries')})", flush=True)

    n_correct = sum(1 for r in results if r.get("correct"))
    n_err = sum(1 for r in results if "error" in r)
    print(f"\n=== SUMMARY: {n_correct}/{len(results)} correct "
          f"({100*n_correct/max(1,len(results)):.1f}%)  errors={n_err} ===")

    if args.save:
        _save(Path(args.save), results, budget, cfg.model, vcfg.model, args.split)
        ck = _ckpt_path(args.save)
        if ck.exists():
            ck.unlink()
        print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
