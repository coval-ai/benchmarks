# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Chat-completions accumulator, generic client, and dental agent tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from coval_bench.llm.dental import load_dental_agent
from coval_bench.llm.openai_compat import (
    AuthError,
    OpenAICompatClient,
    TurnAccumulator,
    UpstreamError,
)


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    return "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish_reason}]})


def test_accumulator_times_the_first_meaningful_delta() -> None:
    turn = TurnAccumulator(10.0)
    turn.feed(_chunk({"role": "assistant"}), 10.1)
    turn.feed(_chunk({"content": "Hello"}), 10.25)
    turn.feed(_chunk({"content": " there"}, "stop"), 10.4)
    result = turn.result(10.5)

    assert result.content == "Hello there"
    assert result.finish_reason == "stop"
    assert result.ttft_ms == pytest.approx(250.0)
    assert result.total_ms == pytest.approx(500.0)


def test_accumulator_reassembles_tool_calls_and_keeps_truncation() -> None:
    turn = TurnAccumulator(5.0)
    first = {"index": 0, "id": "call-1", "function": {"name": "end", "arguments": '{"reason":'}}
    turn.feed(_chunk({"tool_calls": [first]}), 5.2)
    rest = {"index": 0, "function": {"name": "Call", "arguments": '"done"}'}}
    turn.feed(_chunk({"tool_calls": [rest]}, "tool_calls"), 5.3)
    result = turn.result(5.4)

    assert result.finish_reason == "tool_calls"
    assert result.ttft_ms == pytest.approx(200.0)
    assert result.tool_calls == (
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "endCall", "arguments": '{"reason":"done"}'},
        },
    )

    cut = TurnAccumulator(0)
    cut.feed(_chunk({"tool_calls": [first]}, "length"), 0.1)
    assert cut.result(0.2).finish_reason == "length"


def test_accumulator_rejects_errors_empty_and_unfinished_streams() -> None:
    with pytest.raises(UpstreamError, match="generation failed"):
        TurnAccumulator(0).feed('data: {"error":{"message":"generation failed"}}', 0.1)
    with pytest.raises(UpstreamError, match="empty completion"):
        TurnAccumulator(0).result(1)

    turn = TurnAccumulator(0)
    turn.feed(_chunk({"content": "partial"}), 0.1)
    with pytest.raises(UpstreamError, match="before the completion finished"):
        turn.result(0.2)
    turn.feed("data: [DONE]", 0.3)
    assert turn.result(0.4).content == "partial"


@pytest.mark.asyncio
async def test_client_prompts_and_arms_the_model_itself() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=_chunk({"content": "Hi"}, "stop"))

    tools = [{"type": "function", "function": {"name": "book", "parameters": {}}}]
    client = OpenAICompatClient(
        "secret-key",
        "https://llm.test/v1/",
        "gpt-test",
        "Be brief.",
        tools,
        extra_body={"reasoning_effort": "none"},
        transport=httpx.MockTransport(handler),
    )
    try:
        session = await client.create_session()
        result = await client.stream_turn(session.call_id, [{"role": "user", "content": "Hi"}])
    finally:
        await client.aclose()

    assert session.call_id and result.content == "Hi"
    (request,) = requests
    assert request.url == "https://llm.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret-key"
    body = json.loads(request.content)
    assert body["model"] == "gpt-test"
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert body["tools"] == tools
    assert body["stream"] is True
    assert body["reasoning_effort"] == "none"
    assert "secret-key" not in repr(client)


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "error"), [(401, AuthError), (500, UpstreamError)])
async def test_client_classifies_http_errors(status: int, error: type[Exception]) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(status))
    client = OpenAICompatClient("key", "https://llm.test", "m", "", [], transport=transport)
    try:
        with pytest.raises(error):
            await client.stream_turn("call", [])
    finally:
        await client.aclose()


def test_dental_agent_carries_the_contract_prompt_and_tools() -> None:
    agent = load_dental_agent()
    assert "BrightSmile Dental" in agent.system_prompt
    assert "lookup_patient" in {tool["function"]["name"] for tool in agent.tools}
