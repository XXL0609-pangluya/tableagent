"""LLM client over an OpenAI-compatible endpoint.

Phase 0.5 scope: a thin chat() wrapper + a native-tool-calling capability probe.
The full ToolCallProtocol (native vs prompted backends) is built on top of this in
Phase 1; this module deliberately stays provider-agnostic and dependency-light.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from openai import OpenAI

from .config import LLMConfig, load_llm_config
from .schemas import ToolCall


# Module-level pacing shared across ALL clients (solver + verifier hit the same
# endpoint, so spacing must be global). Single-threaded runner → no lock needed.
_last_request_ts: float = 0.0


def _pace(min_interval_s: float) -> None:
    global _last_request_ts
    if min_interval_s <= 0:
        return
    now = time.time()
    wait = _last_request_ts + min_interval_s - now
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.time()


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
        self._client: Optional[OpenAI] = None
        if self.config.protocol == "openai":
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
        if self.config.protocol == "anthropic":
            return self._chat_anthropic(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            )

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

    def _create_with_retry(self, kwargs: dict[str, Any], attempts: int = 6, backoff_s: float = 2.0):
        """Call the endpoint with retries on transient errors (timeouts / 5xx / rate limits).

        A single dropped request used to surface as an empty prediction in a long
        batch run. The company endpoint throttles bursts as `500 - 上游服务错误[429]`,
        so we use exponential backoff with jitter and wait MUCH longer when the error
        looks like rate limiting (429), since the whole server is shedding load.
        """
        last_exc: Optional[Exception] = None
        for i in range(attempts):
            try:
                _pace(self.config.min_interval_s)
                if self._client is None:
                    raise RuntimeError("OpenAI client is not initialized")
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if i >= attempts - 1:
                    break
                msg = str(exc)
                rate_limited = "429" in msg or "rate" in msg.lower() or \
                    type(exc).__name__ == "RateLimitError"
                # exponential backoff (2,4,8,16,32...) with longer base for 429
                base = backoff_s * (4.0 if rate_limited else 1.0)
                delay = min(base * (2 ** i), 60.0) + random.uniform(0, 1.5)
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _anthropic_messages_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/messages"):
            return base
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    @staticmethod
    def _anthropic_text(raw: dict[str, Any]) -> str:
        blocks = raw.get("content") or []
        if not isinstance(blocks, list):
            return ""
        parts: list[str] = []
        for blk in blocks:
            if isinstance(blk, dict) and blk.get("type") == "text":
                txt = blk.get("text")
                if txt:
                    parts.append(str(txt))
        return "".join(parts)

    @staticmethod
    def _to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
        system_parts: list[str] = []
        out: list[dict[str, str]] = []
        for m in messages:
            role = str(m.get("role") or "user")
            content = m.get("content") or ""
            if isinstance(content, list):
                content_text = " ".join(str(x) for x in content)
            else:
                content_text = str(content)
            if role == "system":
                if content_text.strip():
                    system_parts.append(content_text)
                continue
            if role not in ("user", "assistant"):
                # The verifier path only uses system/user. If other roles appear,
                # fold them into a user note so the request still stays valid.
                content_text = f"[{role}] {content_text}"
                role = "user"
            out.append({"role": role, "content": content_text})
        if not out:
            out = [{"role": "user", "content": ""}]
        return "\n\n".join(system_parts).strip(), out

    def _anthropic_create_with_retry(
        self,
        payload: dict[str, Any],
        attempts: int = 6,
        backoff_s: float = 2.0,
    ) -> dict[str, Any]:
        last_exc: Optional[Exception] = None
        data = json.dumps(payload).encode("utf-8")
        for i in range(attempts):
            try:
                _pace(self.config.min_interval_s)
                req = urlrequest.Request(
                    self._anthropic_messages_url(),
                    data=data,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.config.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
                with urlrequest.urlopen(req, timeout=self.config.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                return json.loads(body)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if i >= attempts - 1:
                    break
                msg = str(exc)
                rate_limited = "429" in msg or "rate" in msg.lower()
                if isinstance(exc, urlerror.HTTPError):
                    try:
                        err_body = exc.read().decode("utf-8", errors="ignore")
                        rate_limited = rate_limited or ("429" in err_body)
                    except Exception:  # noqa: BLE001
                        pass
                base = backoff_s * (4.0 if rate_limited else 1.0)
                delay = min(base * (2 ** i), 60.0) + random.uniform(0, 1.5)
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _chat_anthropic(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        model: Optional[str],
    ) -> LLMResponse:
        if tools:
            raise RuntimeError(
                "Anthropic protocol path currently supports verifier-style text calls only "
                "(tool-calling for generator is not implemented yet)."
            )
        system, anthropic_msgs = self._to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": anthropic_msgs,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if system:
            payload["system"] = system
        raw = self._anthropic_create_with_retry(payload)
        usage_raw = raw.get("usage") or {}
        input_toks = int(usage_raw.get("input_tokens") or 0)
        output_toks = int(usage_raw.get("output_tokens") or 0)
        usage = {
            "prompt_tokens": input_toks,
            "completion_tokens": output_toks,
            "total_tokens": input_toks + output_toks,
        }
        return LLMResponse(
            text=self._anthropic_text(raw),
            tool_calls=[],
            finish_reason=str(raw.get("stop_reason") or ""),
            usage=usage,
            raw=raw,
        )

    def probe_native_tools(self) -> tuple[bool, str]:
        """Check whether the model supports native function calling.

        Returns (supported, detail). 'detail' is a short human-readable note.
        """
        if self.config.protocol == "anthropic":
            return False, "anthropic protocol path does not expose native tool-calling in this client"
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
