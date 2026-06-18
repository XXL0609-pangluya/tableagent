"""Print full agent trajectories for the regression and both-wrong buckets.

Run: python -m scripts.inspect_traces <baseline.json> <agent.json> <trace.jsonl> [bucket]
  bucket in {regress, both_wrong, agent_wins}  (default: regress)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, evaluator

SPLIT = "random-split-1-dev"


def load_rows(path):
    with open(path, encoding="utf8") as f:
        return {r["id"]: r for r in json.load(f)["rows"]}


def load_traces(path):
    out = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            out[r["example_id"]] = r
    return out


def correct(targets, ex_id, items):
    if ex_id not in targets:
        return None
    return evaluator.check_denotation(targets[ex_id], evaluator.to_value_list(items) if items else [])


def short(s, n=160):
    s = (s or "").replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + "…"


def main() -> int:
    base = load_rows(sys.argv[1])
    agent = load_rows(sys.argv[2])
    traces = load_traces(sys.argv[3])
    bucket = sys.argv[4] if len(sys.argv) > 4 else "regress"
    targets = evaluator.load_targets_from_tagged(evaluator.find_tagged_path(data.DEFAULT_DATASET_ROOT, SPLIT))

    ids = []
    for i in base:
        if i not in agent:
            continue
        b, a = correct(targets, i, base[i]["pred"]), correct(targets, i, agent[i]["pred"])
        if bucket == "regress" and b and not a:
            ids.append(i)
        elif bucket == "both_wrong" and not b and not a:
            ids.append(i)
        elif bucket == "agent_wins" and a and not b:
            ids.append(i)

    print(f"bucket={bucket}  n={len(ids)}\n")
    for i in ids:
        r = agent[i]
        tr = traces.get(i, {})
        n_runpy = n_runpy_err = 0
        print("=" * 80)
        print(f"[{i}] src={r.get('src')} steps={r.get('steps')}")
        print(f"  Q    : {r['q']}")
        print(f"  gold : {r['gold']}")
        print(f"  base : {base[i]['pred']}")
        print(f"  agent: {r['pred']}")
        for e in tr.get("events", []):
            if e["kind"] != "tool_call":
                continue
            name = e["tool_call"]["name"]
            args = e["tool_call"]["args"]
            obs = (e.get("observation") or {}).get("tool_result", {})
            ok = obs.get("ok")
            payload = obs.get("error") if not ok else obs.get("content_text")
            argstr = short(args.get("code", "") or json.dumps(args, ensure_ascii=False), 90)
            if name == "run_python":
                n_runpy += 1
                if not ok:
                    n_runpy_err += 1
            print(f"    s{e['step']} {name}({argstr}) -> {'OK' if ok else 'ERR'}: {short(payload, 130)}")
        print(f"  >> run_python calls={n_runpy} errors={n_runpy_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
