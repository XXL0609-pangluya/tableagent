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
        return {_to_jsonable(k) if not isinstance(k, str) else k: _to_jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    # numpy / pandas scalars and other exotic leaves: coerce to a JSON-safe form
    # so a stray numpy.int64 in run_python output can never crash trace logging.
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


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
        """Append one JSON record. Never raises into the caller: trace logging must
        not be able to discard a successful prediction (it once did — a stray numpy
        scalar made json.dumps throw and the run loop treated it as a failed example)."""
        record = {
            "trace_id": self.trace_id,
            "example_id": self.example_id,
            "elapsed_ms": round((time.time() - self._t0) * 1000, 1),
            "events": [_to_jsonable(e) for e in self.events],
        }
        if extra:
            record["meta"] = _to_jsonable(extra)
        try:
            line = json.dumps(record, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 — last-resort minimal record
            line = json.dumps({
                "trace_id": self.trace_id,
                "example_id": self.example_id,
                "trace_error": f"{type(exc).__name__}: {exc}",
            }, ensure_ascii=False, default=str)
        try:
            with open(self.out_path, "a", encoding="utf8") as fout:
                fout.write(line + "\n")
        except Exception:  # noqa: BLE001 — disk/logging issues never fail a run
            pass
