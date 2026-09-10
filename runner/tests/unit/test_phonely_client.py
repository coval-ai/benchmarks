# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Phonely streaming client tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from coval_bench.llm.phonely import (
    PhonelyAuthError,
    PhonelyClient,
    PhonelySessionExpired,
    PhonelyUpstreamError,
)


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    return "data: " + json.dumps(
        {
            "id": "chatcmpl-1",
            "created": 123,
            "choices": [{"delta": delta, "finish_reason": finish_reason}],
        }
    )


@pytest.mark.asyncio
async def test_client_uses_the_programmatic_calls_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/calls/session":
            return httpx.Response(200, json={"callId": "call-1", "expiresAt": "later"})
        stream = "\n".join(
            (_chunk({"role": "assistant"}), _chunk({"content": "Hi"}, "stop"), "data: [DONE]")
        )
        return httpx.Response(200, content=stream)

    client = PhonelyClient(
        "secret-key",
        "agent-1",
        "https://phonely.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        session = await client.create_session()
        result = await client.stream_turn(session.call_id, [{"role": "user", "content": "Hi"}])
    finally:
        await client.aclose()

    assert session.call_id == "call-1"
    assert session.expires_at == "later"
    assert result.content == "Hi"
    assert requests[0].headers["X-Authorization"] == "secret-key"
    assert json.loads(requests[0].content) == {"agentId": "agent-1"}
    assert requests[1].headers["Authorization"] == "Bearer secret-key"
    assert requests[1].url.path == "/api/v1/chat/completions"
    assert json.loads(requests[1].content)["stream"] is True
    assert "secret-key" not in repr(client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, PhonelyAuthError),
        (403, PhonelyAuthError),
        (404, PhonelySessionExpired),
        (500, PhonelyUpstreamError),
    ],
)
async def test_client_classifies_http_errors(status: int, error: type[Exception]) -> None:
    client = PhonelyClient(
        "key",
        "agent",
        transport=httpx.MockTransport(lambda _request: httpx.Response(status)),
    )
    try:
        with pytest.raises(error):
            await client.stream_turn("call", [])
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_client_maps_network_failures_to_upstream_errors() -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    client = PhonelyClient("key", "agent", transport=httpx.MockTransport(unreachable))
    try:
        with pytest.raises(PhonelyUpstreamError, match="unreachable"):
            await client.create_session()
    finally:
        await client.aclose()
