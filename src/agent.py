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
from dataclasses import dataclass
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
from .tools.base import ToolRegistry, ToolSpec

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


@dataclass
class Prompts:
    charter: str
    general_skill: str

    @property
    def system(self) -> str:
        return self.charter + "\n\n" + self.general_skill


def load_prompts(prompts_dir: str = _PROMPTS_DIR) -> Prompts:
    with open(os.path.join(prompts_dir, "AGENT.md"), encoding="utf8") as f:
        charter = f.read()
    with open(os.path.join(prompts_dir, "skills", "general.md"), encoding="utf8") as f:
        general = f.read()
    return Prompts(charter=charter, general_skill=general)


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


def run_example(
    example: Example,
    table_context: TableContext,
    registry: ToolRegistry,
    client: LLMClient,
    prompts: Prompts,
    budget: Optional[Budget] = None,
    tracer=None,
) -> Prediction:
    budget = budget or Budget()
    state = AgentState(example=example, table_context=table_context, budget=budget)

    messages: list[dict] = [
        {"role": "system", "content": prompts.system},
        {"role": "user", "content": _build_user(example, table_context)},
    ]
    tools = to_openai_tools(registry.specs(state))

    last_run_items: Optional[list[str]] = None
    last_text: str = ""
    terminated = False

    for step in range(budget.max_steps):
        state.steps_used = step + 1
        # On the final allowed step, stop exploring and force a commitment.
        if step == budget.max_steps - 1:
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
            if result.terminate:
                terminated = True
        if terminated:
            break

    # ---- Finalize (always produce something) ----
    items = state.current_answer
    source = "submit_answer"
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
    evidence.update({"answer_source": source, "steps_used": state.steps_used, "terminated": terminated})
    pred = Prediction(
        id=example.id,
        items=items or [],
        evidence=evidence,
        trace_id=getattr(tracer, "trace_id", None),
    )
    if tracer:
        tracer.add(TraceEvent(step=state.steps_used, kind="final", note=f"items={pred.items} source={source}"))
    return pred
