# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Phonely Programmatic Calls client."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx

from coval_bench.llm.openai_compat import LIMITS, TurnAccumulator
from coval_bench.llm.turn import Session as PhonelySession
from coval_bench.llm.turn import TurnError, TurnResult

if TYPE_CHECKING:
    from coval_bench.config import Settings

_SESSION_TIMEOUT = httpx.Timeout(30.0)


class PhonelyError(TurnError):
    """Base class for failures returned by the Phonely API."""


class PhonelyAuthError(PhonelyError):
    """The configured API key was rejected."""


class PhonelySessionExpired(PhonelyError):
    """The call session is missing or expired."""


class PhonelyUpstreamError(PhonelyError):
    """Phonely failed to produce a usable completion."""


class PhonelyClient:
    """Async client for Phonely's session and streaming chat endpoints."""

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        base_url: str = "https://db.phonely.ai",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._agent_id = agent_id
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(30.0, read=None),
            transport=transport or httpx.AsyncHTTPTransport(http2=True, limits=LIMITS),
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> PhonelyClient | None:
        key = settings.phonely_api_key
        if not (key and key.get_secret_value() and settings.phonely_agent_id):
            return None
        return cls(key.get_secret_value(), settings.phonely_agent_id, settings.phonely_base_url)

    def __repr__(self) -> str:
        return f"PhonelyClient(agent_id={self._agent_id!r})"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_session(self) -> PhonelySession:
        try:
            response = await self._client.post(
                "/api/calls/session",
                headers={"X-Authorization": self._api_key},
                json={"agentId": self._agent_id},
                timeout=_SESSION_TIMEOUT,
            )
        except httpx.RequestError as exc:
            raise PhonelyUpstreamError("Phonely session endpoint is unreachable") from exc
        self._raise_for_status(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise PhonelyUpstreamError("Phonely returned an invalid session response") from exc
        call_id = payload.get("callId") if isinstance(payload, dict) else None
        if not isinstance(call_id, str) or not call_id:
            raise PhonelyUpstreamError("Phonely session response has no callId")
        expires_at = payload.get("expiresAt")
        return PhonelySession(
            call_id=call_id,
            expires_at=expires_at if isinstance(expires_at, str) else None,
        )

    async def stream_turn(self, call_id: str, messages: list[dict[str, Any]]) -> TurnResult:
        started_at = time.perf_counter()
        accumulator = TurnAccumulator(started_at)
        try:
            async with self._client.stream(
                "POST",
                "/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": call_id, "messages": messages, "stream": True},
            ) as response:
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    accumulator.feed(line, time.perf_counter())
        except httpx.RequestError as exc:
            raise PhonelyUpstreamError("Phonely chat endpoint is unreachable") from exc
        return accumulator.result(time.perf_counter())

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        if response.status_code in {401, 403}:
            raise PhonelyAuthError(f"Phonely rejected the request ({response.status_code})")
        if response.status_code == 404:
            raise PhonelySessionExpired("Phonely session is missing or expired")
        raise PhonelyUpstreamError(f"Phonely request failed ({response.status_code})")
