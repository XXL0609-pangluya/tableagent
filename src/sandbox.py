"""Restricted execution of model-generated pandas code (PLAN.md §5).

Phase 0 skeleton: runs code against a DataFrame `df` with a limited builtins set
and a wall-clock timeout, capturing stdout and the structured `answer`/`evidence`
variables that run_python's contract requires.

Security note: this is a soft sandbox (restricted globals + timeout), adequate for
a research setting where we control the prompts. It is NOT a hard security
boundary against adversarial code.
"""
from __future__ import annotations

import contextlib
import io
import signal
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

_ALLOWED_MODULES = {"pandas", "numpy", "re", "math", "datetime", "statistics", "collections"}

_SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
    "len", "list", "map", "max", "min", "range", "round", "set", "sorted", "str",
    "sum", "tuple", "zip", "print", "isinstance", "repr", "reversed", "type",
}


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    answer: Any = None
    evidence: Any = None
    intermediate: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class _Timeout(Exception):
    pass


def _restricted_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in _ALLOWED_MODULES:
        raise ImportError(f"import of '{name}' is not allowed in sandbox")
    return __import__(name, *args, **kwargs)


def run_code(code: str, df: pd.DataFrame, timeout_s: float = 10.0) -> ExecResult:
    """Execute `code` with `df` in scope. Convention: code sets `answer` (and
    optionally `evidence`). Never raises; returns ExecResult."""
    safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                     for k in _SAFE_BUILTINS}
    safe_builtins["__import__"] = _restricted_import

    g: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "df": df,
    }

    stdout = io.StringIO()
    use_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if use_alarm:
        def _handler(signum, frame):
            raise _Timeout()

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)

    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<run_python>", "exec"), g)
        answer = g.get("answer")
        evidence = g.get("evidence")
        intermediate = {
            k: v for k, v in g.items()
            if k not in {"__builtins__", "pd", "df", "answer", "evidence"}
            and not k.startswith("__")
            and isinstance(v, (int, float, str, bool, list, dict))
        }
        return ExecResult(
            ok=True,
            stdout=stdout.getvalue(),
            answer=answer,
            evidence=evidence,
            intermediate=intermediate,
        )
    except _Timeout:
        return ExecResult(ok=False, stdout=stdout.getvalue(), error=f"Execution timed out after {timeout_s}s")
    except Exception as exc:  # noqa: BLE001 - sandbox must capture everything
        return ExecResult(ok=False, stdout=stdout.getvalue(), error=f"{type(exc).__name__}: {exc}")
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
