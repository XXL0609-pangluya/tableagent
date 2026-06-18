"""Connectivity + capability self-check for the configured LLM endpoint.

Run:  python -m scripts.check_llm [model]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_llm_config
from src.llm import LLMClient


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_llm_config(model=model)
    print(f"base_url = {cfg.base_url}")
    print(f"model    = {cfg.model}")
    print(f"api_key  = {cfg.api_key[:6]}...{cfg.api_key[-4:]}")

    client = LLMClient(cfg)

    print("\n[1] Basic chat ...")
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
        )
        print(f"    reply   = {resp.text!r}")
        print(f"    usage   = {resp.usage}")
    except Exception as exc:  # noqa: BLE001
        print(f"    ERROR: {type(exc).__name__}: {exc}")
        return 1

    print("\n[2] Native function-calling probe ...")
    supported, detail = client.probe_native_tools()
    print(f"    native FC supported = {supported}")
    print(f"    detail              = {detail}")

    print("\nCHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
