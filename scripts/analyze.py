"""Unified analysis CLI (wraps compare / trace inspect / live diagnose).

Examples:
  python -m scripts.analyze compare results/baseline_*.json results/agent_*.json
  python -m scripts.analyze traces  results/baseline_*.json results/agent_*.json results/trace_*.jsonl regress
  python -m scripts.analyze diagnose nt-1702 nt-4677
  python -m scripts.analyze diagnose-from results/baseline_*.json results/agent_*.json regress
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(module: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd == "compare":
        return _run("scripts.compare_runs", rest)
    if cmd == "traces":
        return _run("scripts.inspect_traces", rest)
    if cmd == "diagnose":
        return _run("scripts.diagnose", rest)
    if cmd == "diagnose-from":
        return _run("scripts.diagnose", ["--from-compare", *rest])
    print(f"unknown subcommand: {cmd!r}\n")
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
