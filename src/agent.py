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
        r"\brank(ed|ing|s)?\b", r"\bhigher\b", r"\blower\b", r"\blongest\b",
        r"\bshortest\b", r"\bbiggest\b", r"\bsmallest\b", r"\blargest\b",
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
    # Progressive-disclosure modules appended per-question when their regex
    # patterns match the question. Each entry is (name, patterns, text). Empty
    # by default so WTQ (and any caller that doesn't set it) is unaffected. Used
    # by HiTab to route dataset-specific convention blocks by question type
    # instead of dumping the whole skill file on every question.
    dynamic_modules: list[tuple[str, tuple[str, ...], str]] = field(default_factory=list)

    def system_for(self, question: str) -> tuple[str, list[str]]:
        """Assemble the system prompt for a question and report skills used."""
        used = ["general"] + [s for s in select_skills(question) if s in self.skills]
        body = [self.charter] + [self.skills[name] for name in used if name in self.skills]
        q = (question or "").lower()
        for name, patterns, text in self.dynamic_modules:
            if not patterns or any(re.search(p, q) for p in patterns):
                body.append(text)
                used.append(name)
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
    rp_notes = state.evidence.get("run_python_notes") or []
    if isinstance(rp_notes, list) and rp_notes:
        joined = " || ".join(str(x) for x in rp_notes[-5:])
        parts.append(f"run_python_notes: {joined}")
    return " | ".join(parts)[:600] if parts else "(none)"


def _code_signature(code: str) -> str:
    """Normalize run_python code for near-duplicate detection: strip comments,
    blank lines, and whitespace so cosmetic edits don't dodge the check."""
    lines = []
    for raw in (code or "").splitlines():
        line = raw.split("#", 1)[0]            # drop trailing comments
        line = re.sub(r"\s+", "", line)        # collapse all whitespace
        if line:
            lines.append(line)
    return "\n".join(lines).lower()


def _concern_signature(issues: list[str], fix_hint: str) -> str:
    """Normalize verifier concern text for drift detection across rounds."""
    base = " | ".join(issues) if issues else (fix_hint or "")
    base = re.sub(r"\s+", " ", (base or "").strip().lower())
    # keep only semantic signal; drop noisy punctuation
    return re.sub(r"[^a-z0-9 %:/\\-]", "", base)[:280]


def _table_view(tc: TableContext, max_rows: int = 48, max_chars: int = 5200) -> str:
    """Compact-but-broader table rendering for the verifier.

    Progressive browsing needs more than just the first few rows. For long tables,
    show a head+tail slice so the verifier can spot late-table duplicates/boundaries.
    """
    lines = [tc.schema_text, "", "Rows:"]
    df = tc.df
    n = len(df)
    if n <= max_rows:
        idxs = list(range(n))
    else:
        head = max_rows // 2
        tail = max_rows - head
        idxs = list(range(head)) + list(range(max(0, n - tail), n))
    last_i = None
    for i in idxs:
        if last_i is not None and i != last_i + 1:
            lines.append(f"  ... ({i - last_i - 1} rows omitted) ...")
        cells = " | ".join(f"{c}={str(df.iloc[i][c])}" for c in df.columns)
        lines.append(f"  [{i}] {cells}")
        last_i = i
    if n > max_rows:
        lines.append(f"  ... (total rows: {n})")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n…(truncated)"


_BROWSE_HEAVY_RE = re.compile(
    r"\b(how many|number of|count|total|sum|average|difference|most|least|highest|lowest|"
    r"top|maximum|minimum|next|previous|before|after|larger|smaller|higher|lower)\b",
    re.I,
)


def _verification_view(tc: TableContext, question: str, max_chars: int = 6200) -> str:
    """Question-aware table slice for verifier prompts.

    For browse-heavy/count-like questions, include a middle window in addition to
    head/tail so the verifier can see recurring entities not visible in extremes.
    Keep output bounded to avoid token blowups.
    """
    df = tc.df
    n = len(df)
    if n <= 64:
        return _table_view(tc, max_rows=64, max_chars=max_chars)
    if not _BROWSE_HEAVY_RE.search(question or ""):
        return _table_view(tc, max_rows=48, max_chars=max_chars)

    head_n = 20
    mid_n = 24
    tail_n = 20
    mid_start = max(0, (n // 2) - (mid_n // 2))
    mid_end = min(n, mid_start + mid_n)
    idxs = list(range(head_n)) + list(range(mid_start, mid_end)) + list(range(max(0, n - tail_n), n))
    idxs = sorted(set(i for i in idxs if 0 <= i < n))

    lines = [tc.schema_text, "", "Rows:"]
    last_i = None
    for i in idxs:
        if last_i is not None and i != last_i + 1:
            lines.append(f"  ... ({i - last_i - 1} rows omitted) ...")
        cells = " | ".join(f"{c}={str(df.iloc[i][c])}" for c in df.columns)
        lines.append(f"  [{i}] {cells}")
        last_i = i
    lines.append(f"  ... (total rows: {n})")
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
    verify_history: list[dict] = []
    incumbent_candidate: Optional[list[str]] = None
    seen_concern_sigs: set[str] = set()
    last_concern_sig: str = ""
    repeated_concern_rounds = 0
    candidates: list[tuple[list[str], bool]] = []
    drift_guard_enabled = os.environ.get("AGENT_DRIFT_GUARD", "0").strip() == "1"
    # Code history for the auditor: list of (code, output) strings from run_python calls.
    code_history_parts: list[str] = []
    # Anti-spiral: normalized signatures of recent run_python code, to detect when the
    # model re-runs near-identical code without reading the previous output.
    recent_code_sigs: list[str] = []
    # Phase 3b/3c: mutable step ceiling; debate tracking
    effective_max_steps = budget.max_steps
    debate_rounds_granted = 0
    # Multi-round debate: track the concern raised in each round for follow-up context
    last_debate_concern: str = ""
    # How many code_history_parts existed as of the previous verify() call — lets the
    # follow-up audit know whether the generator actually ran NEW code in response to
    # the last concern, vs just resubmitting without engaging (see audit_check).
    code_steps_at_last_verify = 0

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

            spiral_nudge = ""
            if name == "run_python":
                # Anti-spiral: did the model just re-run essentially the same code?
                sig = _code_signature(args.get("code", ""))
                if sig and sig in recent_code_sigs:
                    spiral_nudge = (
                        "\n\n[LOOP DETECTED] You just ran essentially the same code as a "
                        "previous step. Re-running identical code will not change the result. "
                        "STOP and do something different:\n"
                        "  • Read the output/error ABOVE carefully — what did it actually say?\n"
                        "  • If it errored, fix the specific cause (e.g. wrong column name, "
                        "type/encoding, a soft-hyphen or special char in a header).\n"
                        "  • If it returned data, that IS your evidence — interpret it and "
                        "either submit_answer or try a GENUINELY different approach "
                        "(different column, cast types, strip units, inspect raw rows).\n"
                        "  • Do not repeat the same query a third time."
                    )
                if sig:
                    recent_code_sigs.append(sig)

                if result.ok:
                    items_rp = result.structured.get("answer_items")
                    if items_rp:
                        last_run_items = items_rp
                    rp_evidence = result.structured.get("evidence")
                    rp_stdout = (result.structured.get("stdout") or "").strip()
                    note_parts: list[str] = []
                    if rp_evidence:
                        note_parts.append(f"evidence={str(rp_evidence)[:160]}")
                    if items_rp:
                        note_parts.append(f"answer_items={items_rp}")
                    if rp_stdout:
                        one_line = " ".join(rp_stdout.splitlines())
                        note_parts.append(f"stdout={one_line[:160]}")
                    if note_parts:
                        notes = state.evidence.setdefault("run_python_notes", [])
                        if isinstance(notes, list):
                            notes.append(" ; ".join(note_parts))
                            if len(notes) > 8:
                                del notes[:-8]
                    # Record for auditor: code + truncated output
                    code_snippet = args.get("code", "")[:600]
                    output_snippet = (result.content_text or "")[:300]
                    step_label = f"=== run_python step {len(code_history_parts) + 1} ==="
                    code_history_parts.append(
                        f"{step_label}\n{code_snippet}\n--- output ---\n{output_snippet}"
                    )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": (result.content_text or (result.error or "(no output)")) + spiral_nudge,
            })
            if tracer:
                tracer.add(TraceEvent(
                    step=step, kind="tool_call",
                    tool_call=ToolCall(name=name, args=args),
                    observation=Observation(step=step, tool_call=ToolCall(name=name, args=args), tool_result=result),
                ))
            if result.terminate and name == "submit_answer" and result.ok:
                candidate = list(state.current_answer or [])
                if candidate and incumbent_candidate is None:
                    incumbent_candidate = list(candidate)
                code_history = "\n\n".join(code_history_parts)
                new_code_since_last_round = (
                    len(code_history_parts) > code_steps_at_last_verify
                    if verify_retries > 0 else None  # round 1: concept doesn't apply
                )
                vr = verify(
                    verifier_client or client, example.utterance, candidate,
                    df=table_context.df,
                    table_view=_verification_view(table_context, example.utterance),
                    evidence_summary=_evidence_summary(state),
                    code_history=code_history,
                    debate_round=verify_retries + 1,       # round 1 = fresh audit
                    previous_concern=last_debate_concern,  # empty on round 1
                    new_code_since_last_round=new_code_since_last_round,
                )
                code_steps_at_last_verify = len(code_history_parts)
                last_verify = vr.to_dict()
                verify_history.append(last_verify)
                if len(verify_history) > 10:
                    del verify_history[:-10]
                candidates.append((candidate, vr.ok))
                if tracer:
                    tracer.add(TraceEvent(
                        step=step, kind="verify",
                        note=(f"ok={vr.ok} src={vr.source} issues={vr.issues[:2]} "
                              f"resolution={vr.resolution or '-'} "
                              f"new_evidence={'yes' if vr.new_evidence_quote else 'no'} "
                              f"new_code={new_code_since_last_round}"),
                    ))
                if not vr.ok and verify_retries < budget.max_verify_retries:
                    # Record the concern so the next round's follow-up audit can
                    # reference it (multi-round debate memory).
                    current_concern = "; ".join(vr.issues) if vr.issues else (vr.fix_hint or "")
                    concern_sig = _concern_signature(vr.issues, vr.fix_hint)
                    has_new_concern = bool(concern_sig) and concern_sig not in seen_concern_sigs
                    if concern_sig:
                        if concern_sig == last_concern_sig:
                            repeated_concern_rounds += 1
                        else:
                            repeated_concern_rounds = 0
                        last_concern_sig = concern_sig
                        seen_concern_sigs.add(concern_sig)

                    # Drift guard: if concern repeats without new evidence, stop debate
                    # and keep the incumbent instead of spiraling into unstable fallbacks.
                    if drift_guard_enabled and concern_sig and (not has_new_concern) and repeated_concern_rounds >= 1:
                        state.current_answer = list(incumbent_candidate or candidate)
                        terminated = True
                        if tracer:
                            tracer.add(TraceEvent(
                                step=step,
                                kind="debate_stop",
                                note=f"drift_guard: repeated concern without new evidence ({concern_sig[:80]})",
                            ))
                        continue

                    verify_retries += 1
                    # Keep the candidate in the pool; clear current so the model
                    # must resubmit (it may re-confirm the same answer).
                    state.current_answer = None
                    messages[-1]["content"] = (
                        (result.content_text or "")
                        + "\n\n(Verification flagged a concern — see the debate prompt below.)"
                    )
                    # Phase 3b/3c: full debate prompt when audit code is available;
                    # simpler text-only prompt otherwise.
                    if vr.verifier_code:
                        feedback = build_debate_prompt(
                            vr, candidate,
                            debate_round=verify_retries,
                            previous_concern=last_debate_concern,
                        )
                    else:
                        feedback = build_verify_feedback(vr)
                    # Record for next round's follow-up context.
                    last_debate_concern = current_concern
                    messages.append({"role": "user", "content": feedback})
                    # Grant extra steps for this debate round (once per retry).
                    if debate_rounds_granted < verify_retries:
                        effective_max_steps += budget.debate_extra_steps
                        debate_rounds_granted += 1
                        if tracer:
                            tracer.add(TraceEvent(
                                step=step, kind="debate_start",
                                note=f"round={verify_retries} granted +{budget.debate_extra_steps} steps "
                                     f"(effective_max={effective_max_steps}) "
                                     f"concern={current_concern[:80]}",
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

    # ---- Finalize ----
    # Prefer the EARLIEST verified candidate: the first confident answer is
    # more trustworthy than one produced under debate-round pressure.
    # Fallback chain: last clean submit → first verified → first candidate → run_python → forced.
    # NOTE: an empty list ([]) counts as "no answer" — we never want to return empty
    # if the generator ever produced a non-empty answer (empty-answer guardrail).
    items: Optional[list[str]] = None
    source = "submit_answer"
    passed = [c for c, ok in candidates if ok and c]
    if state.current_answer:                       # truthy = non-empty
        items = state.current_answer
    elif passed:
        items, source = passed[0], "first_verified"
    elif drift_guard_enabled and incumbent_candidate:
        items, source = incumbent_candidate, "incumbent_candidate"
    elif any(c for c, _ in candidates):
        items, source = next(c for c, _ in candidates if c), "first_candidate"

    if not items:
        if last_run_items:
            items, source = last_run_items, "last_run_python"
        else:
            forced = _force_final_answer(client, messages, example)
            if forced:
                items, source = forced, "forced_final"
            elif last_text:
                items, source = parse_answer_text(last_text), "last_text"

    # Empty-answer guardrail: if we STILL have nothing, fall back to ANY non-empty
    # answer the generator produced this run rather than submitting [].
    if not items:
        salvage = next((c for c, _ in candidates if c), None) or last_run_items
        if salvage:
            items, source = salvage, "nonempty_salvage"
        else:
            items, source = [], "empty_fallback"

    items = normalize_items(items or [], example.utterance)

    evidence = dict(state.evidence)
    evidence.update({"answer_source": source, "steps_used": state.steps_used,
                     "terminated": terminated, "skills": skills_used,
                     "verify_retries": verify_retries,
                     "candidates": [c for c, _ in candidates],
                     "verify_history": verify_history})
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
