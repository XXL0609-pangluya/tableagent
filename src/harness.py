"""Reliability harness: the tool-call pipeline (PLAN.md §5, §5.1 #1).

`execute_tool` is the single choke point for running a tool: unknown tools,
unavailable tools, missing args, and crashes all become structured ToolResults
fed back to the model — never exceptions that break the loop.
"""
from __future__ import annotations

from typing import Any

from .schemas import AgentState, ToolResult
from .tools.base import ToolRegistry


def execute_tool(
    registry: ToolRegistry,
    name: str,
    args: dict[str, Any],
    state: AgentState,
) -> ToolResult:
    tool = registry.get(name)
    if tool is None:
        names = [t.spec.name for t in registry.all()]
        return ToolResult(ok=False, error=f"Unknown tool {name!r}. Available: {names}")

    reason = tool.available(state)
    if reason is not None:
        return ToolResult(ok=False, error=f"Tool {name!r} is not available: {reason}")

    if not isinstance(args, dict):
        return ToolResult(ok=False, error=f"Tool args must be an object, got {type(args).__name__}")

    required = tool.spec.input_schema.get("required", [])
    missing = [k for k in required if k not in args]
    if missing:
        return ToolResult(ok=False, error=f"Missing required argument(s): {missing}")

    try:
        return tool.run(args, state)
    except Exception as exc:  # noqa: BLE001 - the pipeline must never propagate
        return ToolResult(ok=False, error=f"Tool {name!r} crashed: {type(exc).__name__}: {exc}")
