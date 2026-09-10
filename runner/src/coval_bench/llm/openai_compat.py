# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Streamed chat-completions turns against any OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from coval_bench.llm.turn import Session, TurnError, TurnResult

# Turns are seconds apart; a 5s keepalive would put a TLS handshake inside most TTFTs.
LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=8, keepalive_expiry=300.0)


class UpstreamError(TurnError):
    """The endpoint failed to produce a usable completion."""


class AuthError(UpstreamError):
    """The configured API key was rejected."""


class TurnAccumulator:
    """Reassemble OpenAI SSE chunks while measuring the first meaningful delta."""

    def __init__(self, started_at: float) -> None:
        self._started_at = started_at
        self._first_token_at: float | None = None
        self._content: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._finish_reason: str | None = None
        self._complete = False
        self._output_tokens: int | None = None

    def _stamp_first_token(self, now: float) -> None:
        if self._first_token_at is None:
            self._first_token_at = now

    def feed(self, line: str, now: float) -> None:
        """Consume one SSE line; irrelevant or malformed lines are ignored."""
        line = line.strip()
        if not line.startswith("data:"):
            return
        payload = line.removeprefix("data:").strip()
        if not payload:
            return
        if payload == "[DONE]":
            self._complete = True
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        error = event.get("error")
        if error is not None:
            message = error.get("message", error) if isinstance(error, dict) else error
            raise UpstreamError(f"stream error: {message}")

        usage = event.get("usage")
        if isinstance(usage, dict):
            output_tokens = usage.get("completion_tokens")
            if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
                self._output_tokens = output_tokens

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return
        choice = choices[0]
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                self._stamp_first_token(now)
                self._content.append(content)
            raw_tool_calls = delta.get("tool_calls")
            if isinstance(raw_tool_calls, list) and raw_tool_calls:
                self._stamp_first_token(now)
                self._feed_tool_calls(raw_tool_calls)
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            self._finish_reason = finish_reason
            self._complete = True

    def _feed_tool_calls(self, chunks: list[Any]) -> None:
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            index = chunk.get("index", 0)
            if not isinstance(index, int) or isinstance(index, bool):
                index = 0
            slot = self._tool_calls.setdefault(
                index, {"id": "", "type": "function", "name": "", "arguments": ""}
            )
            call_id = chunk.get("id")
            if isinstance(call_id, str) and call_id:
                slot["id"] = call_id
            call_type = chunk.get("type")
            if isinstance(call_type, str) and call_type:
                slot["type"] = call_type
            function = chunk.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                slot["name"] += name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                slot["arguments"] += arguments

    def result(self, now: float) -> TurnResult:
        """Return the completed turn, rejecting an empty or truncated stream."""
        if self._first_token_at is None:
            raise UpstreamError("the endpoint returned an empty completion")
        if not self._complete:
            raise UpstreamError("the stream ended before the completion finished")
        tool_calls = tuple(
            {
                "id": slot["id"],
                "type": slot["type"],
                "function": {"name": slot["name"], "arguments": slot["arguments"]},
            }
            for _, slot in sorted(self._tool_calls.items())
        )
        finish_reason = self._finish_reason or "stop"
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        return TurnResult(
            content="".join(self._content),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            ttft_ms=(self._first_token_at - self._started_at) * 1000,
            total_ms=(now - self._started_at) * 1000,
            output_tokens=self._output_tokens,
        )


class OpenAICompatClient:
    """A model we prompt ourselves, over the chat-completions wire format.

    Unlike a hosted agent there is no server-side session, so the session id is
    minted here and only threads Coval's turns together.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str,
        tools: list[dict[str, Any]],
        extra_body: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._extra_body = extra_body or {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(30.0, read=None),
            transport=transport or httpx.AsyncHTTPTransport(http2=True, limits=LIMITS),
        )

    def __repr__(self) -> str:
        return f"OpenAICompatClient(model={self._model!r})"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_session(self) -> Session:
        return Session(call_id=uuid.uuid4().hex, expires_at=None)

    async def stream_turn(self, call_id: str, messages: list[dict[str, Any]]) -> TurnResult:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": self._system_prompt}, *messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._extra_body,
        }
        # OpenAI rejects an empty tools array outright.
        if self._tools:
            body["tools"] = self._tools
        started_at = time.perf_counter()
        accumulator = TurnAccumulator(started_at)
        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
            ) as response:
                if response.status_code in {401, 403}:
                    raise AuthError(f"{self._model} rejected the API key ({response.status_code})")
                if response.status_code >= 400:
                    raise UpstreamError(f"{self._model} request failed ({response.status_code})")
                async for line in response.aiter_lines():
                    accumulator.feed(line, time.perf_counter())
        except httpx.RequestError as exc:
            raise UpstreamError(f"{self._model} endpoint is unreachable") from exc
        return accumulator.result(time.perf_counter())
