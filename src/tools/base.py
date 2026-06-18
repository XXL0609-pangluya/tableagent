"""Tool contract and registry (PLAN.md §5.1 #1,#4,#5 + §6).

Design goals:
- Adding a tool = implement Tool + register one line; never touch the agent loop.
- Deterministic plan: unique names asserted at registration.
- Availability gating: a tool can declare it is unusable in the current state,
  and is then hidden (with a diagnostic) rather than silently dropped.
- Errors never escape a tool: run() must return a ToolResult, not raise.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from ..schemas import AgentState, ToolResult


@dataclass
class ToolSpec:
    """What the model is shown about a tool."""

    name: str
    description: str
    # JSON-schema-like dict for arguments (used by both native FC and prompted backends).
    input_schema: dict[str, Any]


class Tool(ABC):
    spec: ToolSpec

    @abstractmethod
    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        """Execute the tool. Must not raise; encode failures in ToolResult."""
        ...

    def available(self, state: AgentState) -> Optional[str]:
        """Return None if usable, else a short diagnostic explaining why hidden."""
        return None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"Duplicate tool name: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def visible(self, state: AgentState) -> tuple[list[Tool], dict[str, str]]:
        """Split into (usable tools, {hidden_name: reason})."""
        visible: list[Tool] = []
        hidden: dict[str, str] = {}
        for tool in self._tools.values():
            reason = tool.available(state)
            if reason is None:
                visible.append(tool)
            else:
                hidden[tool.spec.name] = reason
        return visible, hidden

    def specs(self, state: AgentState) -> list[ToolSpec]:
        visible, _ = self.visible(state)
        return [t.spec for t in visible]
