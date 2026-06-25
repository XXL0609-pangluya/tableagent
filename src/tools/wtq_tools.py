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
import re
from typing import Any

import pandas as pd

from ..context_budget import truncate_text
from ..sandbox import run_code
from ..schemas import AgentState, ToolResult
from .base import Tool, ToolSpec

_SUMMARY_LABELS = {"total", "totals", "sum", "all", "overall", "average",
                   "averages", "subtotal", "grand total", "合计", "总计"}
_PLACEHOLDERS = {"", "-", "–", "—", "n/a", "na", "?", "tbd", "—", "null", "none"}
# leading number, allowing unicode minus (−, U+2212) and thousands separators
_NUM_RE = re.compile(r"^\s*[-−–]?\s*[\d][\d,. ]*")


_MONTHS_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)


def _looks_numeric_dirty(values: list[str]) -> bool:
    """A column that is mostly numbers but wrapped in junk (units, parens, %, unicode
    minus) — needs cleaning before arithmetic. Returns True if worth warning about.

    Excludes date-like columns (most cells contain a month name): those are dates, not
    dirty numerics, and a 'extract the number' hint would mislead."""
    nonblank = [v for v in values if v.strip() and v.strip().lower() not in _PLACEHOLDERS]
    if len(nonblank) < 3:
        return False
    if sum(1 for v in nonblank if _MONTHS_RE.search(v)) >= 0.5 * len(nonblank):
        return False  # date column
    has_num = sum(1 for v in nonblank if _NUM_RE.match(v))
    if has_num < 0.6 * len(nonblank):
        return False
    # "dirty" if a good share carry non-numeric extras (letters/%/parens/unicode minus)
    dirty = sum(1 for v in nonblank
                if re.search(r"[%()a-zA-Z°]", v) or "−" in v or "–" in v)
    return dirty >= max(2, 0.3 * len(nonblank))


def _table_health(df: pd.DataFrame) -> list[str]:
    """Cheap structural 'health report' surfaced up front so the model SEES data traps
    (duplicate split columns, dirty numeric columns, Total rows, multi-line/blank cells)."""
    warns: list[str] = []
    cols = list(df.columns)

    # 1. duplicate/split columns (pandas suffixes dupes as 'X.1', 'X.2')
    split = [c for c in cols if re.match(r"^.+\.\d+$", str(c))
             and re.sub(r"\.\d+$", "", str(c)) in cols]
    if split:
        bases = sorted({re.sub(r"\.\d+$", "", str(c)) for c in split})
        warns.append(
            f"DUPLICATE COLUMNS {split} share base name(s) {bases}: this table is likely "
            "two sub-tables placed SIDE BY SIDE. To count/aggregate a logical column, "
            "combine the duplicates (e.g. pd.concat([df['Position'], df['Position.1']]))."
        )

    # 2. dirty numeric columns
    dirty_cols = []
    for c in cols:
        try:
            vals = df[c].astype(str).tolist()
        except Exception:  # noqa: BLE001
            continue
        if _looks_numeric_dirty(vals):
            sample = next((v for v in vals if v.strip()), "")
            dirty_cols.append(f"{c!r} (e.g. {sample!r})")
    if dirty_cols:
        warns.append(
            "NUMERIC-BUT-DIRTY columns " + ", ".join(dirty_cols[:4]) + ": before any "
            "numeric compare/sum, extract the number with a regex and normalize the minus "
            r"sign, e.g. s = df[col].str.extract(r'([-−–]?[\d.,]+)')[0]"
            ".str.replace('−','-').str.replace(',','').astype(float)."
        )

    # 3. Total / summary rows
    if cols:
        label_col = cols[0]
        try:
            labels = df[label_col].astype(str).str.strip().str.lower()
            tot_rows = df.index[labels.isin(_SUMMARY_LABELS)].tolist()
        except Exception:  # noqa: BLE001
            tot_rows = []
        if tot_rows:
            warns.append(
                f"SUMMARY ROW(S) at index {tot_rows} (label like 'Total'): exclude when "
                "counting/summing data rows; use directly only if the question wants the total."
            )

    # 4. multi-line cells
    multiline = False
    for c in cols:
        try:
            if df[c].astype(str).str.contains("\n").any():
                multiline = True
                break
        except Exception:  # noqa: BLE001
            continue
    if multiline:
        warns.append(
            "MULTI-LINE CELLS present (a cell holds several lines, e.g. 'Name\\n1291–1295'). "
            "The gold answer may be the FULL cell text — keep all parts unless asked otherwise."
        )

    # 5. blank / placeholder cells
    try:
        flat = df.astype(str).to_numpy().ravel()
        n_blank = sum(1 for v in flat if v.strip().lower() in _PLACEHOLDERS)
        if n_blank:
            warns.append(
                f"{n_blank} BLANK/placeholder cell(s) (''/'-'/'–'/'N/A'): handle them "
                "explicitly when filtering or counting (they are not real values)."
            )
    except Exception:  # noqa: BLE001
        pass

    # 6. name variants that would split value_counts / groupby
    for c in cols:
        try:
            vals = [v.strip() for v in df[c].astype(str).tolist() if v.strip()]
        except Exception:  # noqa: BLE001
            continue
        distinct = list(dict.fromkeys(vals))
        if len(distinct) < 5:                       # not an entity-like column
            continue
        if sum(1 for v in distinct if _NUM_RE.match(v)) > len(distinct) * 0.5:
            continue                                # mostly numeric → skip
        low = {v: v.lower() for v in distinct}
        pairs = []
        for a in distinct:
            for b in distinct:
                if a is b:
                    continue
                la, lb = low[a], low[b]
                # b is a strict multi-word extension of a ("Penrhyn Quarry" ⊂
                # "Penrhyn Quarry Railway"): the same entity written two ways.
                if len(la) < len(lb) and lb.startswith(la + " "):
                    pairs.append((a, b))
        if pairs:
            a, b = pairs[0]
            warns.append(
                f"POSSIBLE NAME VARIANTS in column {c!r} (e.g. {a!r} vs {b!r}): "
                "value_counts/groupby would split the same entity across variants. "
                "If you count or rank by this column, normalize variants first "
                "(e.g. map them to a canonical name) and verify by printing the groups."
            )
            break  # one such column is enough signal

    return warns


def _cell_value_set(df: pd.DataFrame) -> set[str]:
    """All distinct cell texts (lower/stripped) for membership checks."""
    cells: set[str] = set()
    for col in df.columns:
        try:
            for v in df[col].astype(str):
                cells.add(v.strip().lower())
        except Exception:  # noqa: BLE001
            continue
    return cells


def split_joined_answer(items: list[str], df: pd.DataFrame) -> list[str]:
    """Fix a multi-value answer wrongly collapsed into ONE delimited string.

    e.g. answer ['1963, 1965'] for a list question becomes ['1963', '1965'].
    High precision: only split a single item when the whole string is NOT itself a
    real cell, but EVERY split piece appears as a complete cell in the table.
    """
    if len(items) != 1:
        return items
    s = items[0].strip()
    if not s:
        return items
    cells = _cell_value_set(df)
    if s.lower() in cells:
        return items  # the joined text is a genuine single cell — keep it
    for delim in (", ", "; ", " and "):
        if delim in s:
            parts = [p.strip() for p in s.split(delim) if p.strip()]
            if len(parts) >= 2 and all(p.lower() in cells for p in parts):
                return parts
    return items


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
        warns = _table_health(tc.df)
        if warns:
            body.append("")
            body.append("⚠ DATA HEALTH NOTES (check these before computing):")
            body += [f"  • {w}" for w in warns]
        text, truncated = truncate_text("\n".join(body), max_chars=3500)
        return ToolResult(
            ok=True,
            content_text=text,
            structured={"columns": tc.columns, "n_rows": tc.n_rows,
                        "health_warnings": warns},
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
        # Repair a multi-value answer collapsed into one delimited string
        # (e.g. ['1963, 1965'] -> ['1963', '1965']) when each piece is a real cell.
        items = split_joined_answer(items, state.table_context.df)
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
