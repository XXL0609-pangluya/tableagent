"""Baselines for comparison. Phase 0.5: direct-answer (no agent, no tools).

Establishes the "no-agent lower bound": dump the (truncated) table + question and
ask the model to answer directly. Everything we build later should beat this.
"""
from __future__ import annotations

from typing import Optional

from .context_budget import truncate_text
from .formatter import parse_answer_text
from .llm import LLMClient
from .schemas import Example, Prediction, TableContext

DIRECT_SYSTEM = (
    "You answer questions about a single data table. "
    "Read the table and output ONLY the final answer, nothing else. "
    "If the answer has multiple items, separate them with ' | '. "
    "Do not explain. Do not add units or words that are not part of the answer."
)


def _render_table(tc: TableContext, max_chars: int = 6000) -> str:
    csv_text = tc.df.to_csv(index=False)
    rendered, _ = truncate_text(csv_text, max_chars=max_chars)
    return rendered


def run_direct_answer(
    example: Example,
    table_context: TableContext,
    client: LLMClient,
) -> tuple[Prediction, dict]:
    """Return (Prediction, usage). Never raises on model/parse errors."""
    table_text = _render_table(table_context)
    user = (
        f"Table ({table_context.n_rows} rows):\n```\n{table_text}\n```\n\n"
        f"Question: {example.utterance}\n\nAnswer:"
    )
    usage: dict = {}
    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=256,
        )
        usage = resp.usage
        items = parse_answer_text(resp.text)
    except Exception as exc:  # noqa: BLE001
        return (
            Prediction(id=example.id, items=[], evidence={"error": f"{type(exc).__name__}: {exc}"}),
            usage,
        )
    return Prediction(id=example.id, items=items, evidence={"raw": resp.text}), usage
