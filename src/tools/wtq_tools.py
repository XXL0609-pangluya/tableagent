"""The WTQ table-QA tool set (PLAN.md §2.1).

Five high-level tools, each with a strong contract and structured feedback:
  - inspect_table   : learn the table's shape/columns/sample values
  - search_columns  : align question words to real headers (column grounding)
  - search_cells    : find cells matching a term (value grounding)
  - run_python      : execute pandas on the table (must set `answer`/`evidence`)
  - submit_answer   : terminate and submit the final answer (+ evidence)

Tools never raise; failures are encoded in ToolResult.
"""
from __future__ import annotations

import difflib
from typing import Any

import pandas as pd

from ..context_budget import truncate_text
from ..sandbox import run_code
from ..schemas import AgentState, ToolResult
from .base import Tool, ToolSpec


def _scalar_str(x: Any) -> str:
    """Render a scalar answer item cleanly (3.0 -> '3')."""
    if isinstance(x, float) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(x).strip()


def coerce_items(value: Any) -> list[str]:
    """Coerce an arbitrary python/pandas value into answer items (list[str])."""
    if value is None:
        return []
    if isinstance(value, pd.Series):
        value = value.tolist()
    elif isinstance(value, pd.DataFrame):
        value = value.to_numpy().flatten().tolist()
    if isinstance(value, (list, tuple, set)):
        return [_scalar_str(v) for v in value if _scalar_str(v) != ""]
    s = _scalar_str(value)
    return [s] if s != "" else []


# ---------------- inspect_table ----------------

class InspectTableTool(Tool):
    spec = ToolSpec(
        name="inspect_table",
        description="Inspect the table: row count, columns with inferred type hints, "
                    "unique-value counts, and a few sample values per column. Call this first.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        tc = state.table_context
        body = [tc.schema_text, "", "First rows:"]
        for row in tc.sample_rows:
            body.append("  " + " | ".join(f"{k}={v!r}" for k, v in row.items()))
        text, truncated = truncate_text("\n".join(body), max_chars=3000)
        return ToolResult(
            ok=True,
            content_text=text,
            structured={"columns": tc.columns, "n_rows": tc.n_rows},
            truncated=truncated,
        )


# ---------------- search_columns ----------------

class SearchColumnsTool(Tool):
    spec = ToolSpec(
        name="search_columns",
        description="Find table columns whose header is similar to a query phrase "
                    "(e.g. 'attendance' -> 'Avg. Attendance'). Use to map question words to real columns.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "word/phrase to match against headers"}},
            "required": ["query"],
        },
    )

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        query = str(args.get("query", "")).strip().lower()
        if not query:
            return ToolResult(ok=False, error="query is required")
        cols = state.table_context.columns
        scored = []
        for col in cols:
            cl = col.lower()
            if query in cl or cl in query:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, query, cl).ratio()
            scored.append((col, round(score, 3)))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:8]
        lines = [f"  {col!r}  (score={s})" for col, s in top]
        return ToolResult(
            ok=True,
            content_text="Columns ranked by similarity to %r:\n%s" % (query, "\n".join(lines)),
            structured={"matches": [{"column": c, "score": s} for c, s in top]},
        )


# ---------------- search_cells ----------------

class SearchCellsTool(Tool):
    spec = ToolSpec(
        name="search_cells",
        description="Find cells whose text contains the query (case-insensitive). "
                    "Returns matching value, row index, and column. Use to ground question terms to real cell values.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "column": {"type": "string", "description": "optional: restrict to this column"},
            },
            "required": ["query"],
        },
    )

    MAX_HITS = 20

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, error="query is required")
        column = args.get("column")
        df = state.table_context.df
        cols = [column] if column else list(df.columns)
        if column and column not in df.columns:
            return ToolResult(
                ok=False,
                error=f"column {column!r} not found. Available: {list(df.columns)}",
            )
        q = query.lower()
        hits = []
        for col in cols:
            mask = df[col].astype(str).str.lower().str.contains(q, regex=False, na=False)
            for idx in df.index[mask]:
                hits.append({"row": int(idx), "column": col, "value": str(df.at[idx, col])})
                if len(hits) >= self.MAX_HITS:
                    break
            if len(hits) >= self.MAX_HITS:
                break
        if not hits:
            return ToolResult(
                ok=True,
                content_text=f"No cells contain {query!r}. Try search_columns or a different term.",
                structured={"hits": []},
            )
        lines = [f"  row={h['row']} col={h['column']!r} value={h['value']!r}" for h in hits]
        return ToolResult(
            ok=True,
            content_text=f"{len(hits)} cell(s) containing {query!r}:\n" + "\n".join(lines),
            structured={"hits": hits},
        )


# ---------------- run_python ----------------

RUN_PYTHON_DESC = (
    "Execute pandas code against the table, available as DataFrame `df` (all columns are strings). "
    "Your code MUST set `answer` to the final answer (a value or a list). "
    "Optionally set `evidence` (e.g. row indices / columns used). "
    "`pd` is available. Cast strings as needed (e.g. remove ',' before int)."
)


class RunPythonTool(Tool):
    spec = ToolSpec(
        name="run_python",
        description=RUN_PYTHON_DESC,
        input_schema={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "python/pandas code that sets `answer`"}},
            "required": ["code"],
        },
    )

    def __init__(self, timeout_s: float = 10.0):
        self.timeout_s = timeout_s

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        code = args.get("code")
        if not code or not str(code).strip():
            return ToolResult(ok=False, error="code is required")
        res = run_code(str(code), state.table_context.df.copy(), timeout_s=self.timeout_s)
        if not res.ok:
            content = f"ERROR: {res.error}"
            if res.stdout:
                content += f"\nstdout:\n{res.stdout}"
            text, _ = truncate_text(content, max_chars=2000)
            return ToolResult(ok=False, content_text=text, error=res.error)

        answer_items = coerce_items(res.answer)
        parts = [f"answer = {res.answer!r}", f"answer_items = {answer_items}"]
        if res.evidence is not None:
            parts.append(f"evidence = {res.evidence!r}")
        if res.stdout.strip():
            parts.append(f"stdout:\n{res.stdout.strip()}")
        if res.answer is None:
            parts.append("WARNING: `answer` was not set. Set `answer` in your code.")
        text, truncated = truncate_text("\n".join(parts), max_chars=2500)
        return ToolResult(
            ok=True,
            content_text=text,
            structured={
                "answer_items": answer_items,
                "evidence": res.evidence,
                "intermediate": res.intermediate,
                "stdout": res.stdout,
            },
            truncated=truncated,
        )


# ---------------- submit_answer ----------------

class SubmitAnswerTool(Tool):
    spec = ToolSpec(
        name="submit_answer",
        description="Submit the FINAL answer. `items` is the list of answer strings "
                    "(one element for a single answer; multiple for a list answer). "
                    "Include `evidence` describing where it came from.",
        input_schema={
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "string", "description": "brief provenance, e.g. rows/columns used"},
            },
            "required": ["items"],
        },
    )

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        items = args.get("items")
        if items is None:
            return ToolResult(ok=False, error="items is required (a list of answer strings)")
        if not isinstance(items, list):
            items = [items]
        items = [str(x).strip() for x in items if str(x).strip() != ""]
        state.current_answer = items
        state.evidence = {"submitted": args.get("evidence")}
        return ToolResult(
            ok=True,
            content_text=f"Answer submitted: {items}",
            structured={"items": items},
            terminate=True,
        )


def build_registry(run_python_timeout_s: float = 10.0):
    """Build the default Phase-1 tool registry."""
    from .base import ToolRegistry

    reg = ToolRegistry()
    reg.register(InspectTableTool())
    reg.register(SearchColumnsTool())
    reg.register(SearchCellsTool())
    reg.register(RunPythonTool(timeout_s=run_python_timeout_s))
    reg.register(SubmitAnswerTool())
    return reg
