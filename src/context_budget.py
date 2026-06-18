"""Token estimation and tool-result truncation (PLAN.md §5.1 #6, #7).

Keeps a single tool result from eating the whole context, and preserves both the
head and the tail (errors/summaries often live at the tail).
"""
from __future__ import annotations

SAFETY_MARGIN = 1.2
_CHARS_PER_TOKEN = 4

try:  # optional, more accurate when available
    import tiktoken  # type: ignore

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback path
    _ENC = None


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (rounded up, with safety margin)."""
    if not text:
        return 0
    if _ENC is not None:
        return int(len(_ENC.encode(text)) * SAFETY_MARGIN)
    return int((len(text) / _CHARS_PER_TOKEN) * SAFETY_MARGIN) + 1


def truncate_text(
    text: str,
    max_chars: int = 8000,
    head_ratio: float = 0.6,
) -> tuple[str, bool]:
    """Head+tail truncation with a middle elision marker.

    Returns (possibly-truncated text, was_truncated).
    """
    if len(text) <= max_chars:
        return text, False
    marker = "\n... [truncated {n} chars] ...\n"
    # Reserve space for the marker (filled in after we know the omitted count).
    budget = max(0, max_chars - len(marker.format(n=999999)))
    head_len = int(budget * head_ratio)
    tail_len = budget - head_len
    head = text[:head_len]
    tail = text[len(text) - tail_len:] if tail_len > 0 else ""
    omitted = len(text) - head_len - tail_len
    # Prefer cutting on newlines for readability.
    nl = head.rfind("\n")
    if nl > head_len * 0.5:
        head = head[: nl + 1]
    return head + marker.format(n=omitted) + tail, True
