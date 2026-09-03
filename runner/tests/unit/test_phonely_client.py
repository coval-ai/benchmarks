# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Phonely streaming client and accumulator tests."""

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
    TurnAccumulator,
)


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    return "data: " + json.dumps(
        {
            "id": "chatcmpl-1",
            "created": 123,
            "choices": [{"delta": delta, "finish_reason": finish_reason}],
        }
    )


def test_accumulator_stamps_the_first_meaningful_content_chunk() -> None:
    turn = TurnAccumulator(10.0)
    turn.feed(_chunk({"role": "assistant"}), 10.1)
    turn.feed(_chunk({"content": "Hello"}), 10.25)
    turn.feed(_chunk({"content": " there"}, "stop"), 10.4)
    result = turn.result(10.5)

    assert result.content == "Hello there"
    assert result.tool_calls == ()
    assert result.finish_reason == "stop"
    assert result.ttft_ms == pytest.approx(250.0)
    assert result.total_ms == pytest.approx(500.0)


def test_accumulator_reassembles_tool_calls_by_index() -> None:
    turn = TurnAccumulator(5.0)
    turn.feed(
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "end", "arguments": '{"reason":'},
                    }
                ]
            }
        ),
        5.2,
    )
    turn.feed(
        _chunk(
            {"tool_calls": [{"index": 0, "function": {"name": "Call", "arguments": '"done"}'}}]},
            "tool_calls",
        ),
        5.3,
    )
    result = turn.result(5.4)

    assert result.content == ""
    assert result.finish_reason == "tool_calls"
    assert result.ttft_ms == pytest.approx(200.0)
    assert result.tool_calls == (
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "endCall", "arguments": '{"reason":"done"}'},
        },
    )


def test_accumulator_rejects_stream_errors_and_empty_completions() -> None:
    with pytest.raises(PhonelyUpstreamError, match="generation failed"):
        TurnAccumulator(0).feed('data: {"error":{"message":"generation failed"}}', 0.1)
    with pytest.raises(PhonelyUpstreamError, match="empty completion"):
        TurnAccumulator(0).result(1)


def test_accumulator_rejects_a_stream_that_ends_before_completion() -> None:
    turn = TurnAccumulator(0)
    turn.feed('data: {"error":null,"choices":[{"delta":{"content":"partial"}}]}', 0.1)
    with pytest.raises(PhonelyUpstreamError, match="before the completion finished"):
        turn.result(0.2)
    turn.feed("data: [DONE]", 0.3)
    assert turn.result(0.4).content == "partial"


def test_accumulator_keeps_a_length_finish_reason_on_truncated_tool_calls() -> None:
    turn = TurnAccumulator(0)
    call = {"index": 0, "id": "c", "function": {"name": "book", "arguments": '{"date": "2026-'}}
    turn.feed(_chunk({"tool_calls": [call]}, "length"), 0.1)
    assert turn.result(0.2).finish_reason == "length"


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
