"""Single-trajectory function-calling agent loop (PLAN.md §2.2, Phase 1).

Flow per question:
  system(charter + skill) + user(question + schema)
  -> [<=budget.max_steps] LLM call with tools
        -> execute each tool call via the harness pipeline
        -> feed structured observations back
  -> submit_answer terminates; otherwise fall back to last run_python / text.

Context note: Phase-1 histories are short (small tables, ~6 steps) so we keep the
full transcript. Per-step context trimming (transformContext) is a Phase-2 refinement.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .formatter import normalize_items, parse_answer_text
from .harness import execute_tool
from .llm import LLMClient
from .schemas import (
    AgentState,
    Budget,
    Example,
    Observation,
    Prediction,
    TableContext,
    ToolCall,
    TraceEvent,
)
from .verifier import build_debate_prompt, build_verify_feedback, verify
from .tools.base import ToolRegistry, ToolSpec

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


# Deterministic skill router: question keyword -> extra skill files to inject on
# top of the always-on `general` skill. Kept simple/auditable (PLAN.md §2.2).
_SKILL_TRIGGERS: dict[str, tuple[str, ...]] = {
    "aggregation": (
        r"\bhow many\b", r"\bnumber of\b", r"\btotal\b", r"\bsum\b", r"\bcombined\b",
        r"\bdifference\b", r"\baverage\b", r"\bmean\b", r"\bcount\b",
        r"\bmost\b", r"\bleast\b", r"\bhighest\b", r"\blowest\b", r"\bfewest\b",
        r"\bgreatest\b", r"\bmaximum\b", r"\bminimum\b", r"\bmore\b", r"\bfewer\b",
    ),
    "positional": (
        r"\bnext\b", r"\bprevious\b", r"\bbefore\b", r"\bafter\b", r"\bpreceding\b",
        r"\bfollowing\b", r"\babove\b", r"\bbelow\b", r"\blast\b", r"\bfirst\b",
        r"\bmiddle\b", r"\bconsecutive\b", r"\bprior\b",
    ),
}


def select_skills(question: str) -> list[str]:
    """Pick extra skills for a question (besides the always-on 'general')."""
    q = (question or "").lower()
    chosen: list[str] = []
    for skill, patterns in _SKILL_TRIGGERS.items():
        if any(re.search(p, q) for p in patterns):
            chosen.append(skill)
    return chosen


@dataclass
class Prompts:
    charter: str
    skills: dict[str, str] = field(default_factory=dict)

    def system_for(self, question: str) -> tuple[str, list[str]]:
        """Assemble the system prompt for a question and report skills used."""
        used = ["general"] + [s for s in select_skills(question) if s in self.skills]
        body = [self.charter] + [self.skills[name] for name in used if name in self.skills]
        return "\n\n".join(body), used


def load_prompts(prompts_dir: str = _PROMPTS_DIR) -> Prompts:
    with open(os.path.join(prompts_dir, "AGENT.md"), encoding="utf8") as f:
        charter = f.read()
    skills: dict[str, str] = {}
    skills_dir = os.path.join(prompts_dir, "skills")
    for name in ("general", "aggregation", "positional"):
        path = os.path.join(skills_dir, f"{name}.md")
        if os.path.exists(path):
            with open(path, encoding="utf8") as f:
                skills[name] = f.read()
    return Prompts(charter=charter, skills=skills)


def to_openai_tools(specs: list[ToolSpec]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": s.name, "description": s.description, "parameters": s.input_schema},
        }
        for s in specs
    ]


def _build_user(example: Example, tc: TableContext) -> str:
    return (
        f"Table schema:\n{tc.schema_text}\n\n"
        f"Question: {example.utterance}\n\n"
        "Find the answer using the tools, then call submit_answer."
    )


_FINAL_INSTRUCTION = (
    "You did not submit an answer and have no steps left. Based ONLY on the tool "
    "outputs above, output the FINAL answer now and nothing else — no explanation, "
    "no code. Rules: copy the value exactly as it appears in the table (full name, "
    "not a code/abbreviation; drop any leading '#'); for a yes/no question answer "
    "'yes' or 'no'; for several values separate them with ' | '."
)


def _force_final_answer(client: LLMClient, messages: list[dict], example: Example) -> list[str]:
    """Last-resort recovery: the answer is often already in the transcript (e.g.
    printed by run_python) but was never submitted. Make one tool-free call to
    extract it instead of returning an empty prediction."""
    try:
        resp = client.chat(
            messages + [{"role": "user", "content": _FINAL_INSTRUCTION}],
            max_tokens=120,
        )
    except Exception:  # noqa: BLE001
        return []
    return parse_answer_text(resp.text or "")


def _evidence_summary(state: AgentState) -> str:
    parts = []
    submitted = state.evidence.get("submitted")
    if submitted:
        parts.append(f"submitted_evidence: {submitted}")
    return " | ".join(parts)[:600] if parts else "(none)"


def _majority_vote(solver_subs: list[list[str]], verifier_recos: list[list[str]]) -> list[str]:
    """Choose the final answer by denotation-clustered majority over solver
    submissions + verifier recomputations.

    Safety property: the verifier can only change the answer when a SOLVER
    submission agrees with it. A lone verifier recompute never wins a tie, so a
    wrong checker (e.g. it recomputed 0) cannot drag a correct answer down. On a
    tie we keep the LATEST solver submission (the deliberately re-derived one)."""
    from .verifier import answers_match

    votes = list(solver_subs) + list(verifier_recos)
    clusters: list[dict] = []  # {rep, count, first_idx}
    for idx, v in enumerate(votes):
        for cl in clusters:
            if answers_match(cl["rep"], v):
                cl["count"] += 1
                break
        else:
            clusters.append({"rep": v, "count": 1, "first_idx": idx})
    if not clusters:
        return []
    top = max(c["count"] for c in clusters)
    tied = [c for c in clusters if c["count"] == top]
    if len(tied) == 1:
        return tied[0]["rep"]
    # tie -> prefer the cluster matching the latest solver submission, else first
    last_solver = solver_subs[-1] if solver_subs else None
    if last_solver is not None:
        for c in tied:
            if answers_match(c["rep"], last_solver):
                return c["rep"]
    tied.sort(key=lambda c: c["first_idx"])
    return tied[0]["rep"]


def _table_view(tc: TableContext, max_rows: int = 30, max_chars: int = 2500) -> str:
    """Compact table rendering for the verifier to inspect (schema + rows)."""
    lines = [tc.schema_text, "", "Rows:"]
    df = tc.df
    for i in range(min(max_rows, len(df))):
        cells = " | ".join(f"{c}={str(df.iloc[i][c])}" for c in df.columns)
        lines.append(f"  [{i}] {cells}")
    if len(df) > max_rows:
        lines.append(f"  ... ({len(df) - max_rows} more rows)")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n…(truncated)"


def run_example(
    example: Example,
    table_context: TableContext,
    registry: ToolRegistry,
    client: LLMClient,
    prompts: Prompts,
    budget: Optional[Budget] = None,
    tracer=None,
    verifier_client: Optional[LLMClient] = None,
) -> Prediction:
    budget = budget or Budget()
    state = AgentState(example=example, table_context=table_context, budget=budget)

    system_prompt, skills_used = prompts.system_for(example.utterance)
    if tracer:
        tracer.add(TraceEvent(step=0, kind="skills", note=",".join(skills_used)))
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user(example, table_context)},
    ]
    tools = to_openai_tools(registry.specs(state))

    last_run_items: Optional[list[str]] = None
    last_text: str = ""
    terminated = False
    verify_retries = 0
    last_verify: Optional[dict] = None
    candidates: list[tuple[list[str], bool]] = []
    verifier_recomputes: list[list[str]] = []
    # Phase 3b: mutable step ceiling — extended when a debate round is triggered.
    # We track how many debate extensions have been granted so we don't keep
    # expanding on every subsequent submit (one extension per verify retry).
    effective_max_steps = budget.max_steps
    debate_rounds_granted = 0

    step = 0
    while step < effective_max_steps:
        state.steps_used = step + 1
        # On the final allowed step, stop exploring and force a commitment.
        if step == effective_max_steps - 1:
            messages.append({
                "role": "user",
                "content": "This is your final step. Call submit_answer now with your best "
                           "answer based on the evidence you already have. Do not call any other tool.",
            })
        try:
            resp = client.chat(messages, tools=tools, tool_choice="auto", max_tokens=900)
        except Exception as exc:  # noqa: BLE001
            if tracer:
                tracer.add(TraceEvent(step=step, kind="llm_error", note=f"{type(exc).__name__}: {exc}"))
            break

        msg = resp.raw.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if tracer:
            tracer.add(TraceEvent(
                step=step, kind="llm_response",
                tokens=resp.usage.get("total_tokens"),
                note=(msg.content or "")[:200],
            ))
        if msg.content:
            last_text = msg.content

        tool_calls = msg.tool_calls or []
        if not tool_calls:
            # Model answered without calling a tool: nudge it to submit explicitly.
            messages.append({
                "role": "user",
                "content": "Do not answer in prose. Call submit_answer(items=[...]) with the final answer.",
            })
            continue

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(registry, name, args, state)

            if name == "run_python" and result.ok:
                items = result.structured.get("answer_items")
                if items:
                    last_run_items = items

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result.content_text or (result.error or "(no output)"),
            })
            if tracer:
                tracer.add(TraceEvent(
                    step=step, kind="tool_call",
                    tool_call=ToolCall(name=name, args=args),
                    observation=Observation(step=step, tool_call=ToolCall(name=name, args=args), tool_result=result),
                ))
            if result.terminate and name == "submit_answer" and result.ok:
                candidate = list(state.current_answer or [])
                vr = verify(
                    verifier_client or client, example.utterance, candidate,
                    df=table_context.df,
                    table_view=_table_view(table_context),
                    evidence_summary=_evidence_summary(state),
                )
                last_verify = vr.to_dict()
                candidates.append((candidate, vr.ok))
                # Only a DISAGREEING recompute becomes a vote (a genuine alternative
                # value). A recompute that merely agrees must not outvote an
                # A/B/C-driven correction where solver and verifier shared a blind spot.
                if vr.compute_match is False and vr.recomputed:
                    verifier_recomputes.append(list(vr.recomputed))
                if tracer:
                    tracer.add(TraceEvent(
                        step=step, kind="verify",
                        note=f"ok={vr.ok} src={vr.source} issues={vr.issues[:3]} hint={vr.fix_hint[:120]}",
                    ))
                if not vr.ok and verify_retries < budget.max_verify_retries:
                    verify_retries += 1
                    # Keep the candidate in the pool; clear current so the model
                    # must resubmit (it may re-confirm the same answer).
                    state.current_answer = None
                    messages[-1]["content"] = (
                        (result.content_text or "")
                        + "\n\n(Verification flagged a concern — see the debate prompt below.)"
                    )
                    # Phase 3b: use the full debate prompt (with verifier code evidence)
                    # when compute evidence is available; simpler text-only prompt otherwise.
                    if vr.verifier_code:
                        feedback = build_debate_prompt(vr, candidate)
                    else:
                        feedback = build_verify_feedback(vr)
                    messages.append({"role": "user", "content": feedback})
                    # Grant extra steps for this debate round (once per retry).
                    if debate_rounds_granted < verify_retries:
                        effective_max_steps += budget.debate_extra_steps
                        debate_rounds_granted += 1
                        if tracer:
                            tracer.add(TraceEvent(
                                step=step, kind="debate_start",
                                note=f"granted +{budget.debate_extra_steps} steps "
                                     f"(effective_max={effective_max_steps})",
                            ))
                else:
                    terminated = True
                    if not vr.ok:
                        # Out of retries and the last answer still failed verify:
                        # don't trust it. Let finalize fall back to the candidate
                        # pool (last passed, else first confident submission).
                        state.current_answer = None
            elif result.terminate:
                terminated = True
        step += 1
        if terminated:
            break

    # ---- Finalize (always produce something) ----
    # Selection policy:
    #  - If compute-verifier produced disagreeing recomputes: MAJORITY VOTE over
    #    {solver submissions} + {disagreeing recomputes}, tie-break to LATEST solver.
    #  - Otherwise: prefer the EARLIEST candidate that passed verification —
    #    the first confident answer is more likely to be correct than one produced
    #    under pressure during a debate round (Phase 3b safety rule).
    #  - Fallback chain: last_run_python → forced_final → last_text → empty.
    items: Optional[list[str]] = None
    source = "submit_answer"
    passed = [c for c, ok in candidates if ok]
    if verifier_recomputes and candidates:
        items = _majority_vote([c for c, _ in candidates], verifier_recomputes)
        source = "vote"
    elif state.current_answer is not None:
        items = state.current_answer  # last submit that terminated cleanly
    elif passed:
        items, source = passed[0], "first_verified"   # earliest verified answer
    elif candidates:
        items, source = candidates[0][0], "first_candidate"

    if items is None:
        if last_run_items:
            items, source = last_run_items, "last_run_python"
        else:
            forced = _force_final_answer(client, messages, example)
            if forced:
                items, source = forced, "forced_final"
            elif last_text:
                items, source = parse_answer_text(last_text), "last_text"
            else:
                items, source = [], "empty_fallback"

    items = normalize_items(items or [], example.utterance)

    evidence = dict(state.evidence)
    evidence.update({"answer_source": source, "steps_used": state.steps_used,
                     "terminated": terminated, "skills": skills_used,
                     "verify_retries": verify_retries,
                     "candidates": [c for c, _ in candidates]})
    if last_verify is not None:
        evidence["verify"] = last_verify
    pred = Prediction(
        id=example.id,
        items=items or [],
        evidence=evidence,
        trace_id=getattr(tracer, "trace_id", None),
    )
    if tracer:
        tracer.add(TraceEvent(step=state.steps_used, kind="final", note=f"items={pred.items} source={source}"))
    return pred
