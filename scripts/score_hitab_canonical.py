"""Report STRICT vs CONVENTION-INVARIANT (canonical) accuracy for HiTab runs.

Runs the canonical scorer (src/hitab_eval_canonical.py) over one or more preds
files and prints how many "errors" were pure format/convention noise vs real
(epistemic) errors. Writes residual_real_error_ids next to each preds file — that
is the clean population for trigger / false-consensus research.

Usage:
  .venv/bin/python scripts/score_hitab_canonical.py <preds.json> [more ...] [--x1000]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hitab_eval_canonical import score  # noqa: E402


def run(preds_path: str, allow_x1000: bool):
    data = json.load(open(preds_path))
    results = data.get("results", data if isinstance(data, list) else [])
    out = score(results, allow_x1000=allow_x1000)
    n = out["n"]
    wrong_strict = n - out["strict_correct"]
    print(f"\n=== {preds_path} ===")
    print(f"  n={n}")
    print(f"  STRICT     : {out['strict_correct']}/{n} = {out['strict_acc']*100:.1f}%   (wrong={wrong_strict})")
    print(f"  CANONICAL  : {out['canonical_correct']}/{n} = {out['canonical_acc']*100:.1f}%   "
          f"(x1000={'on' if allow_x1000 else 'off'})")
    print(f"  recovered by convention : {out['recovered_by_convention']}  "
          f"({out['recovered_by_convention']/max(wrong_strict,1)*100:.0f}% of strict errors were format noise)")
    print(f"  residual REAL errors    : {out['residual_real_errors']}  "
          f"({out['residual_real_errors']/max(wrong_strict,1)*100:.0f}% of strict errors)")
    base = Path(preds_path).with_name(Path(preds_path).stem + "_canonical")
    json.dump(out, open(str(base) + ".json", "w"), indent=2, ensure_ascii=False)
    json.dump(out["residual_ids"],
              open(str(base.with_name(base.stem + "_residual_real_error_ids")) + ".json", "w"),
              indent=2)
    print(f"  wrote {base}.json (+ residual_real_error_ids.json)")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--x1000"]
    allow_x1000 = "--x1000" in sys.argv
    if not args:
        print(__doc__)
        sys.exit(1)
    for p in args:
        run(p, allow_x1000)
