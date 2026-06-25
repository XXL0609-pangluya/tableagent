"""Dataset loading: WTQ data/*.tsv questions and csv/*.csv tables.

Tables are read with all columns as strings (object dtype). WTQ cells are dirty
(commas in numbers, "Did not qualify", parenthetical notes, quoted line breaks),
so we defer all casting to the agent's generated pandas code.
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional

import pandas as pd

from .evaluator import tsv_unescape, tsv_unescape_list
from .schemas import Example, TableContext

# Default dataset root relative to repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DATASET_ROOT = os.path.join(_REPO_ROOT, "WikiTableQuestions")
DEFAULT_DISPUTED_PATH = os.path.join(_REPO_ROOT, "eval", "disputed.json")


def load_disputed(path: str = DEFAULT_DISPUTED_PATH) -> dict[str, str]:
    """Load ids of examples with ambiguous/inconsistent gold answers (id -> reason).

    These are excluded from 'adjusted' accuracy so a buggy benchmark item doesn't
    penalise the system. Returns {} if the file is absent."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf8") as f:
        return dict(json.load(f).get("disputed", {}))


def load_examples(split_basename: str, dataset_root: str = DEFAULT_DATASET_ROOT) -> list[Example]:
    """Load examples from data/<split_basename>.tsv."""
    tsv_path = os.path.join(dataset_root, "data", split_basename + ".tsv")
    examples: list[Example] = []
    with open(tsv_path, "r", encoding="utf8") as fin:
        header = fin.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in fin:
            cols = line.rstrip("\n").split("\t")
            examples.append(
                Example(
                    id=cols[idx["id"]],
                    utterance=cols[idx["utterance"]],
                    table_path=cols[idx["context"]],
                    target_value=tsv_unescape_list(cols[idx["targetValue"]]),
                )
            )
    return examples


def _build_schema_text(df: pd.DataFrame, max_uniques: int = 3) -> str:
    """Compact column summary for prompts: name, inferred dtype hint, sample values."""
    lines = [f"{len(df)} rows x {len(df.columns)} columns"]
    for col in df.columns:
        series = df[col]
        uniques = series.dropna().unique().tolist()
        n_unique = len(uniques)
        sample = ", ".join(repr(str(v)) for v in uniques[:max_uniques])
        lines.append(f"- {col!r} ({n_unique} unique) e.g. {sample}")
    return "\n".join(lines)


def _dedupe_columns(cols: list[str]) -> list[str]:
    """Make column names unique (WTQ tables may repeat headers). Mirrors pandas'
    'X', 'X.1', 'X.2' scheme so duplicate-header tables stay usable."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def _read_table_tsv(abs_tsv_path: str) -> pd.DataFrame:
    """Parse WTQ's tab-separated table variant.

    The .tsv variant is unambiguous: tab-separated fields, embedded newlines/pipes
    escaped (\\n, \\p, \\\\). This avoids the CSV variant's backslash-escaped quotes
    (e.g. \"Popper\") that break RFC-4180 parsers like pandas.
    """
    with open(abs_tsv_path, "r", encoding="utf8") as fin:
        header = [tsv_unescape(h) for h in fin.readline().rstrip("\n").split("\t")]
        rows: list[list[str]] = []
        n = len(header)
        for line in fin:
            cells = [tsv_unescape(c) for c in line.rstrip("\n").split("\t")]
            if len(cells) < n:
                cells += [""] * (n - len(cells))
            elif len(cells) > n:
                cells = cells[:n]
            rows.append(cells)
    return pd.DataFrame(rows, columns=header, dtype=str)


def load_table(table_path: str, dataset_root: str = DEFAULT_DATASET_ROOT) -> TableContext:
    """Load a single table as an all-string DataFrame.

    Prefers the robust .tsv variant; falls back to the .csv with backslash escaping.
    """
    abs_csv = os.path.join(dataset_root, table_path)
    abs_tsv = os.path.splitext(abs_csv)[0] + ".tsv"
    if os.path.exists(abs_tsv):
        df = _read_table_tsv(abs_tsv)
    else:
        df = pd.read_csv(
            abs_csv, dtype=str, keep_default_na=False, na_values=[],
            engine="python", escapechar="\\",
        )
    df.columns = _dedupe_columns([str(c) for c in df.columns])

    n_sample = min(3, len(df))
    sample_rows = [
        {col: str(df.iloc[i][col]) for col in df.columns} for i in range(n_sample)
    ]
    return TableContext(
        table_path=table_path,
        df=df,
        columns=list(df.columns),
        dtypes={col: str(df[col].dtype) for col in df.columns},
        n_rows=len(df),
        schema_text=_build_schema_text(df),
        sample_rows=sample_rows,
    )


def sample_examples(
    examples: list[Example], n: int, seed: int = 13
) -> list[Example]:
    """Deterministic random subset (the 'quick set' for fast iteration)."""
    if n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    return rng.sample(examples, n)


# Evaluation subset seeds (each holdout excludes all prior subsets).
QUICK_SEED = 13
QUICK_SET_SIZE = 200
FRESH_SEED = 29
HOLDOUT_SEED = 41
# Larger mining set for error attribution (run entirely on current code).
MINE_SEED = 57
MINE_SET_SIZE = 1000


def _subset_ids(examples: list[Example], which: str) -> set[str]:
    if which == "quick":
        return {e.id for e in sample_examples(examples, QUICK_SET_SIZE, QUICK_SEED)}
    if which == "fresh":
        quick = _subset_ids(examples, "quick")
        pool = [e for e in examples if e.id not in quick]
        return {e.id for e in sample_examples(pool, QUICK_SET_SIZE, FRESH_SEED)}
    if which == "holdout":
        used = _subset_ids(examples, "quick") | _subset_ids(examples, "fresh")
        pool = [e for e in examples if e.id not in used]
        return {e.id for e in sample_examples(pool, QUICK_SET_SIZE, HOLDOUT_SEED)}
    if which == "mine":
        # Deterministic 1000-example mining set (may overlap the 200-sets; that is
        # fine — used for broad error attribution, not as a disjoint holdout).
        return {e.id for e in sample_examples(examples, MINE_SET_SIZE, MINE_SEED)}
    raise ValueError(f"unknown eval subset: {which!r}")


def eval_subset(
    examples: list[Example], n: int, which: str = "quick"
) -> list[Example]:
    """Return an evaluation subset.

    which="quick":   original quick set (seed=13).
    which="fresh":   disjoint from quick (seed=29).
    which="holdout": disjoint from quick+fresh (seed=41) — use for new test runs.
    """
    ids = _subset_ids(examples, which)
    ordered = [e for e in examples if e.id in ids]
    return ordered[:n]
