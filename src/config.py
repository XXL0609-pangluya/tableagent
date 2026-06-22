"""Centralized configuration (PLAN.md §6: config-driven, no magic constants).

Secrets come from .env (gitignored) via python-dotenv. Nothing here is hardcoded
with a real key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 60.0


def load_llm_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> LLMConfig:
    base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("LLM_API_KEY", "")
    if not base_url or not api_key:
        raise RuntimeError(
            "LLM_BASE_URL / LLM_API_KEY missing. Copy .env.example to .env and fill it in."
        )
    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model or os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        temperature=0.0 if temperature is None else temperature,
    )


def load_verifier_config(model: Optional[str] = None) -> LLMConfig:
    """Config for the verifier — deliberately a DIFFERENT model than the solver
    (industry practice: an independent checker avoids same-model blind spots).

    Falls back to the solver's endpoint/key but a different model. Override via
    VERIFIER_MODEL / VERIFIER_BASE_URL / VERIFIER_API_KEY in .env.
    """
    base_url = os.environ.get("VERIFIER_BASE_URL", os.environ.get("LLM_BASE_URL", "")).rstrip("/")
    api_key = os.environ.get("VERIFIER_API_KEY", os.environ.get("LLM_API_KEY", ""))
    solver_model = os.environ.get("LLM_MODEL", "")
    verifier_model = (
        model
        or os.environ.get("VERIFIER_MODEL")
        # default to the "other" cheap model so it differs from the solver
        or ("deepseek-v4-flash" if solver_model != "deepseek-v4-flash" else "qwen3.6-35b-a3b")
    )
    if not base_url or not api_key:
        raise RuntimeError("VERIFIER/LLM base_url or api_key missing in .env")
    return LLMConfig(base_url=base_url, api_key=api_key, model=verifier_model, temperature=0.0)
