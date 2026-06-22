"""Smoke-test the independent verifier on A/B/C representative cases.

Confirms: (1) the verifier model is reachable and DIFFERENT from the solver,
(2) it returns sane JSON, (3) it can flag a deliberately wrong answer using the
table view. Run: python scripts/check_verifier.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src import data
from src.agent import _table_view
from src.config import load_llm_config, load_verifier_config
from src.llm import LLMClient
from src.verifier import verify

SPLIT = "random-split-1-dev"


def main() -> int:
    solver = load_llm_config()
    vcfg = load_verifier_config()
    print(f"solver model   = {solver.model}")
    print(f"verifier model = {vcfg.model}")
    if vcfg.model == solver.model:
        print("WARNING: verifier and solver use the SAME model; set VERIFIER_MODEL in .env")
    vclient = LLMClient(vcfg)

    examples = data.load_examples(SPLIT, data.DEFAULT_DATASET_ROOT)
    by_id = {ex.id: ex for ex in examples}

    # (good_answer, deliberately_wrong_answer) probes
    probes = [
        ("nt-1470", None),  # disputed — just observe
    ]
    # fall back to first example if specific ids absent
    if "nt-1470" not in by_id:
        probes = [(examples[0].id, None)]

    for ex_id, _ in probes:
        ex = by_id.get(ex_id)
        if ex is None:
            print(f"  (skip {ex_id}: not in split)")
            continue
        tc = data.load_table(ex.table_path, data.DEFAULT_DATASET_ROOT)
        tv = _table_view(tc)
        print(f"\n=== {ex.id} ===")
        print(f"Q: {ex.utterance}")
        print(f"gold: {ex.target_value}")
        vr = verify(vclient, ex.utterance, list(ex.target_value), table_view=tv)
        print(f"verify(gold) -> ok={vr.ok} axis={vr.axis} src={vr.source} issues={vr.issues}")
        vr2 = verify(vclient, ex.utterance, ["__definitely_wrong__"], table_view=tv)
        print(f"verify(wrong) -> ok={vr2.ok} axis={vr2.axis} src={vr2.source} issues={vr2.issues}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
