"""Reusable helpers for post-run error analysis (trace deep-dives, bucket compare).

Used by scripts/diagnose.py, scripts/inspect_traces.py, and scripts/compare_runs.py
so we don't duplicate trace parsing / trajectory formatting logic.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from . import data, evaluator
from .schemas import Example, Prediction, TableContext, TraceEvent

SPLIT = "random-split-1-dev"


def short_text(s: Any, n: int = 160) -> str:
    s = ("" if s is None else str(s)).replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + "…"


def load_run_rows(path: str) -> dict[str, dict]:
    with open(path, encoding="utf8") as f:
        return {r["id"]: r for r in json.load(f)["rows"]}


def load_traces_jsonl(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            out[r["example_id"]] = r
    return out


def load_gold_targets() -> dict[str, list]:
    tagged = evaluator.find_tagged_path(data.DEFAULT_DATASET_ROOT, SPLIT)
    return evaluator.load_targets_from_tagged(tagged)


def is_correct(targets: dict, ex_id: str, items: list[str]) -> Optional[bool]:
    if ex_id not in targets:
        return None
    pred_vals = evaluator.to_value_list(items) if items else []
    return evaluator.check_denotation(targets[ex_id], pred_vals)


def compare_buckets(
    baseline_rows: dict[str, dict],
    agent_rows: dict[str, dict],
    targets: Optional[dict] = None,
) -> dict[str, list[str]]:
    """Partition example ids into both_right / both_wrong / agent_wins / regress."""
    targets = targets or load_gold_targets()
    ids = [i for i in baseline_rows if i in agent_rows]
    buckets: dict[str, list[str]] = {
        "both_right": [], "both_wrong": [], "agent_wins": [], "regress": [],
    }
    for ex_id in ids:
        b = is_correct(targets, ex_id, baseline_rows[ex_id]["pred"])
        a = is_correct(targets, ex_id, agent_rows[ex_id]["pred"])
        if b and a:
            buckets["both_right"].append(ex_id)
        elif not b and not a:
            buckets["both_wrong"].append(ex_id)
        elif a and not b:
            buckets["agent_wins"].append(ex_id)
        else:
            buckets["regress"].append(ex_id)
    return buckets


def format_tool_event(ev: dict | TraceEvent, code_max: int = 500, obs_max: int = 500) -> list[str]:
    """Render one tool_call trace event as printable lines."""
    if isinstance(ev, TraceEvent):
        ev = {
            "step": ev.step,
            "kind": ev.kind,
            "tool_call": {"name": ev.tool_call.name, "args": ev.tool_call.args} if ev.tool_call else None,
            "observation": {
                "tool_result": {
                    "ok": ev.observation.tool_result.ok,
                    "content_text": ev.observation.tool_result.content_text,
                    "error": ev.observation.tool_result.error,
                }
            } if ev.observation else None,
        }
    if ev.get("kind") != "tool_call":
        return []
    tc = ev["tool_call"] or {}
    name = tc.get("name", "?")
    args = tc.get("args") or {}
    obs = (ev.get("observation") or {}).get("tool_result") or {}
    ok = obs.get("ok")
    payload = obs.get("error") if not ok else obs.get("content_text")
    lines: list[str] = []
    if name == "run_python":
        lines.append(f"  s{ev['step']} run_python:")
        for ln in str(args.get("code", "")).splitlines():
            lines.append(f"      | {ln}")
        lines.append(f"      -> {'OK' if ok else 'ERR'}: {short_text(payload, code_max)}")
    else:
        argstr = short_text(args.get("code", "") or json.dumps(args, ensure_ascii=False), 120)
        lines.append(f"  s{ev['step']} {name}({argstr}) -> {short_text(payload, obs_max)}")
    return lines


def format_trajectory(events: list[dict | TraceEvent]) -> str:
    lines: list[str] = []
    for ev in events:
        lines.extend(format_tool_event(ev))
    return "\n".join(lines)


def format_example_report(
    ex: Example,
    tc: TableContext,
    pred: Prediction,
    events: list[dict | TraceEvent],
    *,
    table_max: int = 1400,
) -> str:
    """Full deep-dive block: question, table snippet, gold vs pred, trajectory."""
    parts = [
        "=" * 90,
        f"[{ex.id}]  Q: {ex.utterance}",
        f"  gold={ex.target_value}  ->  agent={pred.items}  "
        f"(src={pred.evidence.get('answer_source')}, skills={pred.evidence.get('skills')})",
    ]
    if pred.evidence.get("verify"):
        parts.append(f"  verify={pred.evidence.get('verify')}")
    parts.extend([
        f"  columns: {list(tc.df.columns)}",
        "  --- TABLE ---",
        "  " + short_text(tc.df.to_string(), table_max).replace(" ⏎ ", "\n  "),
        "  --- TRAJECTORY ---",
        format_trajectory(events) or "  (no tool calls)",
        "",
    ])
    return "\n".join(parts)


def format_bucket_from_runs(
    baseline_rows: dict[str, dict],
    agent_rows: dict[str, dict],
    traces: dict[str, dict],
    bucket: str,
    targets: Optional[dict] = None,
) -> str:
    """Print regress/agent_wins/both_wrong with compact trajectories from saved traces."""
    buckets = compare_buckets(baseline_rows, agent_rows, targets)
    ids = buckets.get(bucket, [])
    lines = [f"bucket={bucket}  n={len(ids)}\n"]
    for ex_id in ids:
        r = agent_rows[ex_id]
        tr = traces.get(ex_id, {})
        n_runpy = n_runpy_err = 0
        lines.append("=" * 80)
        lines.append(f"[{ex_id}] src={r.get('src')} steps={r.get('steps')}")
        if r.get("verify"):
            lines.append(f"  verify={r.get('verify')}")
        lines.append(f"  Q    : {r['q']}")
        lines.append(f"  gold : {r['gold']}")
        lines.append(f"  base : {baseline_rows[ex_id]['pred']}")
        lines.append(f"  agent: {r['pred']}")
        for ev in tr.get("events", []):
            if ev.get("kind") != "tool_call":
                continue
            name = ev["tool_call"]["name"]
            if name == "run_python":
                n_runpy += 1
                obs = (ev.get("observation") or {}).get("tool_result", {})
                if not obs.get("ok"):
                    n_runpy_err += 1
            lines.extend(format_tool_event(ev, code_max=90, obs_max=130))
        lines.append(f"  >> run_python calls={n_runpy} errors={n_runpy_err}")
    return "\n".join(lines)
