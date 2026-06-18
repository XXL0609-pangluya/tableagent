"""LLM client over an OpenAI-compatible endpoint.

Phase 0.5 scope: a thin chat() wrapper + a native-tool-calling capability probe.
The full ToolCallProtocol (native vs prompted backends) is built on top of this in
Phase 1; this module deliberately stays provider-agnostic and dependency-light.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from .config import LLMConfig, load_llm_config
from .schemas import ToolCall


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_llm_config()
        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_s,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        resp = self._create_with_retry(kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__raw__": tc.function.arguments}
            tool_calls.append(ToolCall(name=tc.function.name, args=args))

        usage = {}
        if resp.usage is not None:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
            raw=resp,
        )

    def _create_with_retry(self, kwargs: dict[str, Any], attempts: int = 3, backoff_s: float = 1.5):
        """Call the endpoint with retries on transient errors (timeouts / 5xx / rate limits).

        A single dropped request used to surface as an empty prediction in a long
        batch run, so we retry a few times before giving up.
        """
        last_exc: Optional[Exception] = None
        for i in range(attempts):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if i < attempts - 1:
                    time.sleep(backoff_s * (i + 1))
        raise last_exc  # type: ignore[misc]

    def probe_native_tools(self) -> tuple[bool, str]:
        """Check whether the model supports native function calling.

        Returns (supported, detail). 'detail' is a short human-readable note.
        """
        probe_tool = [{
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a value back.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }]
        try:
            resp = self.chat(
                messages=[{"role": "user", "content": "Call the echo tool with value='hi'."}],
                tools=probe_tool,
                tool_choice="auto",
                max_tokens=128,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"native tools call raised: {type(exc).__name__}: {exc}"
        if resp.tool_calls:
            return True, f"emitted tool_call: {resp.tool_calls[0].name}"
        return False, "no tool_calls returned (will fall back to prompted backend)"
