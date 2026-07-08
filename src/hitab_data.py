"""HiTab dataset loading + hierarchical-table flattening (HiTab-specific).

HiTab tables are hierarchical matrices, NOT flat CSVs like WTQ, so this module is
kept separate from `data.py` (which stays WTQ-only and untouched). Naming is
prefixed `hitab_` per the project convention: dataset-specific code is labelled so
it is obvious at a glance.

Raw table JSON (datasets/HiTab/data/tables/raw/<table_id>.json) fields we use:
  - title:                    caption string
  - texts:                    2D grid (list[rows] of list[str] cells)
  - top_header_rows_num:      #rows at the top that form the (multi-level) column header
  - left_header_columns_num:  #columns at the left that form the (multi-level) row header
  - merged_regions:           [{first_row,last_row,first_column,last_column}, ...]
                              (a spanned cell: only the top-left anchor holds text,
                               the rest are '' and must be filled from the anchor)

Flattening strategy (Option 1 — faithful matrix):
  * Fill merged regions from their anchor, then forward-fill any remaining blanks in
    the top-header rows (horizontally) and left-header columns (vertically) — this is
    the standard HiTab de-spanning heuristic.
  * Data columns (col >= LH) get a hierarchical name = join of the top-header cells,
    e.g. "chart positions | us".
  * Left-header columns (col < LH) become ordinary string key-columns, named from the
    top-left corner label (e.g. "year", "title") or "row_header_{i}" as fallback.
  * Rows below the top header become DataFrame rows (all cells kept as strings, matching
    the WTQ convention so the agent's pandas code does its own casting).

This yields a plain all-string DataFrame that the existing dataset-agnostic tools
(inspect_table / search_columns / search_cells / run_python) and `run_example` can
consume without modification.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

from .schemas import Example, TableContext

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
HITAB_ROOT = os.environ.get(
    "HITAB_DATASET_ROOT",
    os.path.join(_REPO_ROOT, "datasets", "HiTab"),
)
HITAB_DATA_DIR = os.path.join(HITAB_ROOT, "data")
HITAB_RAW_TABLE_DIR = os.path.join(HITAB_DATA_DIR, "tables", "raw")

# HiTab uses table_id as the identifier; we synthesise a table_path for TableContext.
_SPLIT_FILES = {
    "train": "train_samples.jsonl",
    "dev": "dev_samples.jsonl",
    "test": "test_samples.jsonl",
}


# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------

def _answer_to_items(answer: list) -> list[str]:
    """HiTab gold answers are list[float|int|str]; render to strings the way the
    agent would (3.0 -> '3') so downstream matching is uniform."""
    items: list[str] = []
    for x in answer:
        if isinstance(x, float) and abs(x - round(x)) < 1e-9:
            items.append(str(int(round(x))))
        else:
            items.append(str(x).strip())
    return items


def _question_to_str(question) -> str:
    """A handful of HiTab samples store `question` as a LIST (e.g. ["2016"])
    instead of a string. Downstream code calls `.lower()` on the utterance, so a
    list crashes the whole example ('list' object has no attribute 'lower').
    Coerce to a plain string here so loading is robust to that data anomaly."""
    if isinstance(question, list):
        return " ".join(str(x).strip() for x in question if str(x).strip())
    return str(question)


def load_hitab_examples(split: str = "dev", data_dir: str = HITAB_DATA_DIR) -> list[Example]:
    """Load HiTab examples from data/<split>_samples.jsonl.

    Maps HiTab fields to the shared Example contract:
      id            <- sample id
      utterance     <- question
      table_path    <- table_id (resolved by load_hitab_table)
      target_value  <- answer rendered to list[str]
    """
    fname = _SPLIT_FILES.get(split)
    if fname is None:
        raise ValueError(f"unknown HiTab split {split!r} (use train/dev/test)")
    path = os.path.join(data_dir, fname)
    examples: list[Example] = []
    with open(path, encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            examples.append(
                Example(
                    id=s["id"],
                    utterance=_question_to_str(s["question"]),
                    table_path=s["table_id"],
                    target_value=_answer_to_items(s.get("answer", [])),
                )
            )
    return examples


# --------------------------------------------------------------------------
# Table flattening
# --------------------------------------------------------------------------

def _fill_merged(grid: list[list[str]], merged_regions: list[dict]) -> list[list[str]]:
    """Copy each merged region's anchor (top-left) value into the spanned blanks."""
    g = [list(row) for row in grid]
    for reg in merged_regions:
        r0, r1 = reg["first_row"], reg["last_row"]
        c0, c1 = reg["first_column"], reg["last_column"]
        if r0 >= len(g) or c0 >= len(g[r0]):
            continue
        anchor = g[r0][c0]
        for r in range(r0, min(r1, len(g) - 1) + 1):
            for c in range(c0, c1 + 1):
                if r < len(g) and c < len(g[r]) and str(g[r][c]).strip() == "":
                    g[r][c] = anchor
    return g


def _rectangular(grid: list[list[str]]) -> list[list[str]]:
    """Pad ragged rows so every row has the same width."""
    width = max((len(r) for r in grid), default=0)
    out = []
    for r in grid:
        r = [("" if v is None else str(v)) for v in r]
        if len(r) < width:
            r = r + [""] * (width - len(r))
        out.append(r)
    return out


# HiTab writes the literal string "none" in the top-left corner cell when a
# header has no label (it is a placeholder, NOT a real value). Letting it leak
# into column names produces confusing names like "none | year". Treat it as
# blank when building header names.
_HEADER_PLACEHOLDERS = {"", "none", "<top>", "<left>", "null"}


def _dedupe_join(parts: list[str]) -> str:
    """Join header levels top->bottom, dropping blanks / placeholders and
    consecutive duplicates."""
    seen: list[str] = []
    for p in parts:
        p = str(p).strip()
        if p.lower() in _HEADER_PLACEHOLDERS:
            continue
        if seen and seen[-1].lower() == p.lower():
            continue
        seen.append(p)
    return " | ".join(seen)


def _dedupe_columns(cols: list[str]) -> list[str]:
    """Ensure unique column names (mirror pandas 'X', 'X.1' scheme)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        c = c or "col"
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def _min_row_index(tree: Optional[dict]) -> Optional[int]:
    """Smallest data row index referenced by a header tree (authoritative header
    extent). left_root's node row_indexes point at the first DATA row."""
    if not tree:
        return None
    idxs: list[int] = []

    def walk(n: dict) -> None:
        ri = n.get("row_index", -1)
        if isinstance(ri, int) and ri >= 0:
            idxs.append(ri)
        for c in n.get("children", []) or []:
            walk(c)

    walk(tree)
    return min(idxs) if idxs else None


def _min_col_index(tree: Optional[dict]) -> Optional[int]:
    if not tree:
        return None
    idxs: list[int] = []

    def walk(n: dict) -> None:
        ci = n.get("column_index", -1)
        if isinstance(ci, int) and ci >= 0:
            idxs.append(ci)
        for c in n.get("children", []) or []:
            walk(c)

    walk(tree)
    return min(idxs) if idxs else None


def _header_extents(raw: dict, n_rows: int, n_cols: int) -> tuple[int, int]:
    """Return (th, lh): number of top-header rows and left-header columns.

    HiTab's raw `top_header_rows_num` is unreliable (systematically over-counts by
    ~1 on ToTTo-derived tables). The `top_root` / `left_root` trees are authoritative
    (HiTab's own model code uses them): the smallest data-row index in `left_root`
    is where data starts (= #top-header rows); likewise `top_root` for columns.
    Fall back to the raw counts only when a tree is missing/empty.
    """
    th = _min_row_index(raw.get("left_root"))
    lh = _min_col_index(raw.get("top_root"))
    if th is None:
        th = int(raw.get("top_header_rows_num", 1) or 1)
    if lh is None:
        lh = int(raw.get("left_header_columns_num", 1) or 1)
    th = max(1, min(th, n_rows))
    lh = max(0, min(lh, n_cols))
    return th, lh


def flatten_hitab_table(raw: dict) -> tuple[pd.DataFrame, dict]:
    """Flatten a HiTab raw hierarchical table into an all-string DataFrame.

    Returns (df, meta) where meta carries title + header dimensions for prompts.
    """
    grid = _rectangular(raw.get("texts", []))
    if not grid:
        return pd.DataFrame(), {"title": raw.get("title", ""), "top_header_rows_num": 0,
                                "left_header_columns_num": 0}
    n_rows = len(grid)
    n_cols = len(grid[0])
    th, lh = _header_extents(raw, n_rows, n_cols)

    grid = _fill_merged(grid, raw.get("merged_regions", []))

    # Forward-fill top-header rows horizontally (spanned column groups).
    for r in range(th):
        last = ""
        for c in range(n_cols):
            v = str(grid[r][c]).strip()
            if v == "":
                grid[r][c] = last
            else:
                last = v

    # Forward-fill left-header columns vertically (spanned row groups) over data rows.
    for c in range(lh):
        last = ""
        for r in range(th, n_rows):
            v = str(grid[r][c]).strip()
            if v == "":
                grid[r][c] = last
            else:
                last = v

    # Build column names.
    col_names: list[str] = []
    for c in range(n_cols):
        if c < lh:
            # left-header column: name from the top-left corner label if present
            corner = _dedupe_join([grid[r][c] for r in range(th)])
            col_names.append(corner or f"row_header_{c}")
        else:
            name = _dedupe_join([grid[r][c] for r in range(th)])
            col_names.append(name or f"col_{c}")
    col_names = _dedupe_columns(col_names)

    # Data rows.
    data_rows = []
    for r in range(th, n_rows):
        data_rows.append([str(grid[r][c]) for c in range(n_cols)])

    df = pd.DataFrame(data_rows, columns=col_names, dtype=str)
    meta = {
        "title": raw.get("title", ""),
        "top_header_rows_num": th,
        "left_header_columns_num": lh,
        "n_grid_rows": n_rows,
        "n_grid_cols": n_cols,
    }
    return df, meta


def _build_schema_text(df: pd.DataFrame, meta: dict, max_uniques: int = 3) -> str:
    """Compact column summary for prompts, prefixed with the table title (HiTab
    questions frequently depend on the caption for context)."""
    lines = []
    title = meta.get("title")
    if title:
        lines.append(f"Table title: {title}")
    lines.append(f"{len(df)} data rows x {len(df.columns)} columns "
                 f"(hierarchical: {meta.get('top_header_rows_num')} top-header row(s), "
                 f"{meta.get('left_header_columns_num')} left-header column(s))")
    for col in df.columns:
        series = df[col]
        uniques = series.dropna().unique().tolist()
        sample = ", ".join(repr(str(v)) for v in uniques[:max_uniques])
        lines.append(f"- {col!r} ({len(uniques)} unique) e.g. {sample}")
    return "\n".join(lines)


def load_hitab_table(table_id: str, raw_dir: str = HITAB_RAW_TABLE_DIR) -> TableContext:
    """Load + flatten one HiTab table into a TableContext (all-string DataFrame)."""
    path = os.path.join(raw_dir, table_id + ".json")
    with open(path, encoding="utf8") as f:
        raw = json.load(f)
    df, meta = flatten_hitab_table(raw)

    n_sample = min(3, len(df))
    sample_rows = [
        {col: str(df.iloc[i][col]) for col in df.columns} for i in range(n_sample)
    ]
    return TableContext(
        table_path=table_id,
        df=df,
        columns=list(df.columns),
        dtypes={col: str(df[col].dtype) for col in df.columns},
        n_rows=len(df),
        schema_text=_build_schema_text(df, meta),
        sample_rows=sample_rows,
    )
