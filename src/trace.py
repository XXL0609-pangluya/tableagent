"""Structured, append-only trace logging (PLAN.md §5 'full-chain trace').

Every run writes one JSONL file. This is the lifeline for debugging an
AI-developed agent and for error attribution.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
import uuid
from typing import Any, Optional

from .schemas import TraceEvent


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class Tracer:
    """Collects TraceEvents for one example and appends a single JSON record."""

    def __init__(self, out_path: str, example_id: str):
        self.out_path = out_path
        self.example_id = example_id
        self.trace_id = uuid.uuid4().hex
        self.events: list[TraceEvent] = []
        self._t0 = time.time()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def add(self, event: TraceEvent) -> None:
        self.events.append(event)

    def flush(self, extra: Optional[dict] = None) -> None:
        record = {
            "trace_id": self.trace_id,
            "example_id": self.example_id,
            "elapsed_ms": round((time.time() - self._t0) * 1000, 1),
            "events": [_to_jsonable(e) for e in self.events],
        }
        if extra:
            record["meta"] = _to_jsonable(extra)
        with open(self.out_path, "a", encoding="utf8") as fout:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
