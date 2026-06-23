"""Core data contracts for the whole system (see PLAN.md §2.4).

These are intentionally defined before any logic so every module builds against
stable seams. Use plain dataclasses here (no heavy deps); pydantic is reserved
for tool-argument validation from Phase 1 onward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class Example:
    """One WTQ example loaded from a data/*.tsv line."""

    id: str
    utterance: str
    table_path: str  # relative path like "csv/204-csv/590.csv"
    target_value: list[str]  # gold answer items (already TSV-unescaped)


@dataclass
class TableContext:
    """A loaded table plus cheap derived metadata used for prompt assembly."""

    table_path: str
    df: pd.DataFrame
    columns: list[str]
    dtypes: dict[str, str]
    n_rows: int
    schema_text: str  # compact, human/LLM-readable column summary
    sample_rows: list[dict[str, str]]  # first few rows as plain dicts


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Uniform result for every tool. Errors never raise out of a tool;
    they are encoded here and fed back to the model (see PLAN.md §5/§5.1)."""

    ok: bool
    content_text: str = ""  # what the model sees (already size-governed)
    structured: dict[str, Any] = field(default_factory=dict)  # e.g. run_python answer/evidence
    error: Optional[str] = None
    truncated: bool = False
    terminate: bool = False  # set by submit_answer to end the agent loop


@dataclass
class Observation:
    """A (tool_call, tool_result) pair anchored to a loop step."""

    step: int
    tool_call: ToolCall
    tool_result: ToolResult


@dataclass
class Budget:
    max_steps: int = 6
    max_retries: int = 2
    max_verify_retries: int = 1  # Phase 2b: one correction pass after failed verify
    debate_extra_steps: int = 3  # Phase 3b: extra steps granted per debate round
    max_tokens: Optional[int] = None
    step_timeout_s: float = 10.0


@dataclass
class AgentState:
    """Mutable per-question state. Durable facts are kept separate from the
    ephemeral message scratchpad (State vs Scratchpad, PLAN.md §2.4.1)."""

    example: Example
    table_context: TableContext
    observations: list[Observation] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)  # reduced, durable findings
    attempts: int = 0
    current_answer: Optional[list[str]] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    steps_used: int = 0


@dataclass
class Prediction:
    """Final output for one example. Always carries provenance for error
    attribution (PLAN.md §2.4)."""

    id: str
    items: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    trace_id: Optional[str] = None


@dataclass
class TraceEvent:
    step: int
    kind: str  # e.g. "tool_call", "llm_request", "verify", "fallback"
    prompt_hash: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    observation: Optional[Observation] = None
    tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    note: Optional[str] = None
