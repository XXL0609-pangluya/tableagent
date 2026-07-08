"""Build false-consensus report from HiTab preds + traces.

Outputs:
  - JSON metrics summary
  - Markdown report (human-readable)

Main quantities (strict + canonical):
  - 2x2 table: debate-triggered vs correctness
  - false-consensus share among all errors
  - trigger rate / subgroup accuracies
  - candidate diversity and oracle-selector gap

Usage:
  .venv/bin/python scripts/false_consensus_report.py \
    --preds results/HiTab/<run>/preds.json \
    --trace-dir results/HiTab/<run>/traces
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.hitab_eval import _norm_str, answer_match
from src.hitab_eval_canonical import canonical_match

# Parses the "verify" TraceEvent.note string written by src/agent.py:
#   "ok=False src=audit issues=[...] resolution=insufficient new_evidence=yes new_code=True"
_VERIFY_NOTE_RE = re.compile(
    r"ok=(?P<ok>True|False)\s+src=(?P<src>\S+)\s+issues=.*?"
    r"resolution=(?P<resolution>\S+)\s+new_evidence=(?P<new_evidence>yes|no)\s+new_code=(?P<new_code>True|False|None)"
)


def _sig(items: list[str]) -> tuple[str, ...]:
    vals = [str(x) for x in (items or []) if str(x).strip() != ""]
    return tuple(sorted(_norm_str(v) for v in vals))


def _read_trace(path: Path) -> dict:
    raw = path.read_text(encoding="utf8").splitlines()
    if not raw:
        return {}
    obj = json.loads(raw[0])
    evs = obj.get("events", [])
    debate = any((e.get("kind") or e.get("type")) == "debate_start" for e in evs)
    submits: list[list[str]] = []
    verify_rounds: list[dict] = []
    for e in evs:
        kind = e.get("kind") or e.get("type")
        tc = e.get("tool_call") or {}
        if tc.get("name") == "submit_answer":
            items = tc.get("args", {}).get("items") or []
            if items:
                submits.append([str(x) for x in items if str(x).strip() != ""])
        if kind == "verify":
            m = _VERIFY_NOTE_RE.search(e.get("note") or "")
            if m:
                verify_rounds.append(
                    {
                        "ok": m.group("ok") == "True",
                        "resolution": m.group("resolution"),
                        "new_evidence": m.group("new_evidence") == "yes",
                        "new_code": m.group("new_code"),  # "True"/"False"/"None" (str)
                    }
                )
    return {
        "id": obj.get("example_id"),
        "debate": debate,
        "submissions": submits,
        "distinct_candidates": len({_sig(s) for s in submits}),
        "verify_rounds": verify_rounds,
    }


def _debate_engagement(traces: dict) -> dict:
    """Anchored-rejection diagnostic (proposed fix #3).

    Looks only at FOLLOW-UP audit rounds (round >= 2, i.e. every verify_rounds
    entry after the first per trace) where the trace was produced with the
    resolution-tagging verifier (src/verifier.py audit_check). Buckets:
      - accepted            : auditor agreed the concern was resolved
      - accepted_forced     : anti-anchoring guard fired (generator gave new
                               evidence, auditor could not cite/refute it ->
                               mechanically forced to accept)
      - unaddressed         : generator did NOT run new code; auditor is
                               legitimately entitled to keep the concern open
      - insufficient/new_issue : auditor engaged with new evidence and still
                               rejected (may be correct — needs eyeballing)
      - untagged            : round predates this fix / resolution missing
    """
    counts: Counter = Counter()
    traces_with_forced_accept = 0
    traces_with_engaged_reject = 0
    total_followup_rounds = 0
    for t in traces.values():
        rounds = t.get("verify_rounds") or []
        followups = rounds[1:]  # round 1 is the fresh audit, not a follow-up
        if not followups:
            continue
        saw_forced = False
        saw_engaged_reject = False
        for r in followups:
            total_followup_rounds += 1
            res = r["resolution"] if r["resolution"] and r["resolution"] != "-" else "untagged"
            counts[res] += 1
            if res == "accepted_forced":
                saw_forced = True
            if res in ("insufficient", "new_issue"):
                saw_engaged_reject = True
        if saw_forced:
            traces_with_forced_accept += 1
        if saw_engaged_reject:
            traces_with_engaged_reject += 1
    return {
        "total_followup_rounds": total_followup_rounds,
        "resolution_counts": dict(counts),
        "traces_with_anchored_rejection_prevented": traces_with_forced_accept,
        "traces_with_engaged_rejection": traces_with_engaged_reject,
        "anchored_rejection_prevented_rate": (
            counts["accepted_forced"] / total_followup_rounds if total_followup_rounds else 0.0
        ),
        "legitimate_persistence_rate": (
            counts["unaddressed"] / total_followup_rounds if total_followup_rounds else 0.0
        ),
    }


def _oracle_hit(submissions: list[list[str]], gold: list[str], canonical: bool) -> bool:
    for s in submissions:
        if canonical:
            if canonical_match(s, gold):
                return True
        else:
            if answer_match(s, gold):
                return True
    return False


def _two_by_two(rows: list[dict], correctness_key: str) -> dict:
    cnt = Counter()
    for r in rows:
        cnt[(bool(r["debate"]), bool(r[correctness_key]))] += 1
    nd_right = cnt[(False, True)]
    nd_wrong = cnt[(False, False)]
    d_right = cnt[(True, True)]
    d_wrong = cnt[(True, False)]
    n = nd_right + nd_wrong + d_right + d_wrong
    wrong = nd_wrong + d_wrong
    return {
        "n": n,
        "trigger_rate": (d_right + d_wrong) / n if n else 0.0,
        "no_debate": {"correct": nd_right, "wrong": nd_wrong},
        "debate": {"correct": d_right, "wrong": d_wrong},
        "overall_acc": (nd_right + d_right) / n if n else 0.0,
        "no_debate_acc": nd_right / (nd_right + nd_wrong) if (nd_right + nd_wrong) else 0.0,
        "debate_acc": d_right / (d_right + d_wrong) if (d_right + d_wrong) else 0.0,
        "false_consensus_wrong": nd_wrong,
        "false_consensus_share_of_errors": nd_wrong / wrong if wrong else 0.0,
    }


def _diversity_and_oracle(rows: list[dict], correctness_key: str, canonical: bool) -> dict:
    n = len(rows)
    debate_rows = [r for r in rows if r["debate"]]
    multi_all = sum(1 for r in rows if r["distinct_candidates"] > 1)
    multi_debate = sum(1 for r in debate_rows if r["distinct_candidates"] > 1)
    current_correct_all = sum(1 for r in rows if r[correctness_key])
    current_correct_debate = sum(1 for r in debate_rows if r[correctness_key])
    oracle_all = sum(1 for r in rows if _oracle_hit(r["submissions"], r["gold"], canonical))
    oracle_debate = sum(1 for r in debate_rows if _oracle_hit(r["submissions"], r["gold"], canonical))
    n_debate = len(debate_rows)
    return {
        "overall": {
            "n": n,
            "multi_candidate_rate": multi_all / n if n else 0.0,
            "current_acc": current_correct_all / n if n else 0.0,
            "oracle_acc": oracle_all / n if n else 0.0,
            "oracle_gap_pt": ((oracle_all - current_correct_all) / n * 100.0) if n else 0.0,
        },
        "debate_subgroup": {
            "n": n_debate,
            "multi_candidate_rate": multi_debate / n_debate if n_debate else 0.0,
            "current_acc": current_correct_debate / n_debate if n_debate else 0.0,
            "oracle_acc": oracle_debate / n_debate if n_debate else 0.0,
            "oracle_gap_pt": ((oracle_debate - current_correct_debate) / n_debate * 100.0) if n_debate else 0.0,
        },
    }


def _format_md(summary: dict) -> str:
    s2 = summary["strict_2x2"]
    c2 = summary["canonical_2x2"]
    sdiv = summary["strict_diversity_oracle"]
    cdiv = summary["canonical_diversity_oracle"]
    de = summary.get("debate_engagement", {})
    rc = de.get("resolution_counts", {})
    return "\n".join(
        [
            "# False Consensus Report",
            "",
            f"- run: `{summary['run_label']}`",
            f"- preds: `{summary['preds_path']}`",
            f"- trace_dir: `{summary['trace_dir']}`",
            f"- matched_examples: **{summary['matched_examples']}**",
            "",
            "## 2x2 (STRICT)",
            f"- no debate: correct={s2['no_debate']['correct']} wrong={s2['no_debate']['wrong']}",
            f"- debate: correct={s2['debate']['correct']} wrong={s2['debate']['wrong']}",
            f"- trigger_rate={s2['trigger_rate']*100:.1f}%  overall_acc={s2['overall_acc']*100:.1f}%",
            f"- false_consensus_share_of_errors={s2['false_consensus_share_of_errors']*100:.1f}%",
            "",
            "## 2x2 (CANONICAL)",
            f"- no debate: correct={c2['no_debate']['correct']} wrong={c2['no_debate']['wrong']}",
            f"- debate: correct={c2['debate']['correct']} wrong={c2['debate']['wrong']}",
            f"- trigger_rate={c2['trigger_rate']*100:.1f}%  overall_acc={c2['overall_acc']*100:.1f}%",
            f"- false_consensus_share_of_errors={c2['false_consensus_share_of_errors']*100:.1f}%",
            "",
            "## Diversity + Oracle (STRICT)",
            f"- overall multi_candidate_rate={sdiv['overall']['multi_candidate_rate']*100:.1f}%  oracle_gap={sdiv['overall']['oracle_gap_pt']:.2f}pt",
            f"- debate  multi_candidate_rate={sdiv['debate_subgroup']['multi_candidate_rate']*100:.1f}%  oracle_gap={sdiv['debate_subgroup']['oracle_gap_pt']:.2f}pt",
            "",
            "## Diversity + Oracle (CANONICAL)",
            f"- overall multi_candidate_rate={cdiv['overall']['multi_candidate_rate']*100:.1f}%  oracle_gap={cdiv['overall']['oracle_gap_pt']:.2f}pt",
            f"- debate  multi_candidate_rate={cdiv['debate_subgroup']['multi_candidate_rate']*100:.1f}%  oracle_gap={cdiv['debate_subgroup']['oracle_gap_pt']:.2f}pt",
            "",
            "## Debate Engagement / Anchored Rejection (fix #2+#3 diagnostic)",
            f"- follow-up audit rounds analyzed: **{de.get('total_followup_rounds', 0)}**",
            f"- resolution breakdown: {rc}",
            f"- traces where the anti-anchoring guard fired (forced accept): "
            f"**{de.get('traces_with_anchored_rejection_prevented', 0)}** "
            f"({de.get('anchored_rejection_prevented_rate', 0)*100:.1f}% of follow-up rounds)",
            f"- traces with legitimate persistence (generator gave no new code, "
            f"verifier kept the concern open): rate={de.get('legitimate_persistence_rate', 0)*100:.1f}%",
            f"- traces where the verifier engaged new evidence and STILL rejected "
            f"(insufficient/new_issue — needs eyeballing): "
            f"**{de.get('traces_with_engaged_rejection', 0)}**",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to preds.json (or preds.json.partial.json)")
    ap.add_argument("--trace-dir", required=True, help="Trace directory with *.jsonl")
    ap.add_argument("--out-prefix", default="", help="Output prefix path (without extension)")
    ap.add_argument("--run-label", default="", help="Optional run label in report")
    args = ap.parse_args()

    preds_path = Path(args.preds)
    trace_dir = Path(args.trace_dir)
    data = json.loads(preds_path.read_text(encoding="utf8"))
    results = data.get("results", data if isinstance(data, list) else [])

    by_id: dict[str, dict] = {}
    for r in results:
        rid = r.get("id")
        if not rid:
            continue
        pred = [str(x) for x in (r.get("pred") or []) if str(x).strip() != ""]
        gold = [str(x) for x in (r.get("gold") or []) if str(x).strip() != ""]
        by_id[rid] = {
            "id": rid,
            "pred": pred,
            "gold": gold,
            "strict_correct": bool(r.get("correct", answer_match(pred, gold))),
            "canonical_correct": bool(canonical_match(pred, gold)),
        }

    traces = {}
    for p in sorted(trace_dir.glob("*.jsonl")):
        row = _read_trace(p)
        if row.get("id"):
            traces[row["id"]] = row

    rows: list[dict] = []
    for rid, r in by_id.items():
        if rid not in traces:
            continue
        t = traces[rid]
        rows.append(
            {
                "id": rid,
                "gold": r["gold"],
                "pred": r["pred"],
                "strict_correct": r["strict_correct"],
                "canonical_correct": r["canonical_correct"],
                "debate": bool(t["debate"]),
                "submissions": t["submissions"],
                "distinct_candidates": t["distinct_candidates"],
            }
        )

    strict_2x2 = _two_by_two(rows, "strict_correct")
    canonical_2x2 = _two_by_two(rows, "canonical_correct")
    strict_div = _diversity_and_oracle(rows, "strict_correct", canonical=False)
    canonical_div = _diversity_and_oracle(rows, "canonical_correct", canonical=True)
    debate_engagement = _debate_engagement(traces)

    summary = {
        "run_label": args.run_label or preds_path.parent.name,
        "preds_path": str(preds_path),
        "trace_dir": str(trace_dir),
        "matched_examples": len(rows),
        "strict_2x2": strict_2x2,
        "canonical_2x2": canonical_2x2,
        "strict_diversity_oracle": strict_div,
        "canonical_diversity_oracle": canonical_div,
        "debate_engagement": debate_engagement,
    }

    if args.out_prefix:
        base = Path(args.out_prefix)
    else:
        base = preds_path.parent / "false_consensus_report"
    json_path = Path(str(base) + ".json")
    md_path = Path(str(base) + ".md")

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    md_path.write_text(_format_md(summary), encoding="utf8")

    print(f"[false-consensus] matched={len(rows)}")
    print(
        f"[strict]    trigger={strict_2x2['trigger_rate']*100:.1f}% "
        f"false_consensus={strict_2x2['false_consensus_share_of_errors']*100:.1f}%"
    )
    print(
        f"[canonical] trigger={canonical_2x2['trigger_rate']*100:.1f}% "
        f"false_consensus={canonical_2x2['false_consensus_share_of_errors']*100:.1f}%"
    )
    print(f"[write] {json_path}")
    print(f"[write] {md_path}")


if __name__ == "__main__":
    main()
