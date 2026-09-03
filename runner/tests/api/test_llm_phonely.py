# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The authenticated Phonely LLM proxy."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import psycopg
import psycopg.rows
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import SecretStr

from coval_bench.llm.phonely import PhonelyClient
from tests.api.conftest import LLM_PROXY_KEY, _make_db_url

AUTH = {"Authorization": f"Bearer {LLM_PROXY_KEY}"}
Handler = Callable[[httpx.Request], httpx.Response]


def _sse(*deltas: dict[str, Any]) -> bytes:
    lines = [
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "created": 123,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )
        for delta in deltas
    ]
    lines.extend(
        [
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
    )
    return "\n\n".join(lines).encode()


@pytest_asyncio.fixture
async def bind_phonely(app: FastAPI) -> AsyncIterator[Callable[[Handler], None]]:
    handlers: list[Handler] = []
    client = PhonelyClient(
        "upstream-key",
        "agent-1",
        "https://phonely.test",
        transport=httpx.MockTransport(lambda request: handlers[-1](request)),
    )
    app.state.phonely_client = client
    yield handlers.append
    await client.aclose()


async def _turn_rows(postgresql: Any) -> list[dict[str, Any]]:
    connection = await psycopg.AsyncConnection.connect(
        _make_db_url(postgresql), autocommit=True, row_factory=psycopg.rows.dict_row
    )
    try:
        cursor = await connection.execute(
            "SELECT * FROM benchmarks_v2.llm_turns ORDER BY created_at, id"
        )
        return list(await cursor.fetchall())
    finally:
        await connection.close()


async def test_proxy_auth_configuration_and_route_location(
    client: AsyncClient, app: FastAPI
) -> None:
    configured_client = app.state.phonely_client
    assert isinstance(configured_client, PhonelyClient)
    assert (await client.post("/llm/phonely/session", json={})).status_code == 401
    assert (
        await client.post(
            "/llm/phonely/session", json={}, headers={"Authorization": "Bearer wrong"}
        )
    ).status_code == 401
    assert (await client.post("/v1/llm/phonely/session", json={}, headers=AUTH)).status_code == 404

    app.state.phonely_client = None
    assert (await client.post("/llm/phonely/session", json={}, headers=AUTH)).status_code == 503
    lowercase = {"Authorization": f"bearer {LLM_PROXY_KEY}"}
    lowercase_response = await client.post("/llm/phonely/session", json={}, headers=lowercase)
    assert lowercase_response.status_code == 503
    app.state.phonely_client = configured_client
    settings = app.state.settings
    app.state.settings = settings.model_copy(update={"llm_proxy_secret": SecretStr("")})
    empty = {"Authorization": "Bearer "}
    assert (await client.post("/llm/phonely/session", json={}, headers=empty)).status_code == 401
    app.state.settings = settings.model_copy(update={"llm_proxy_secret": None})
    assert (await client.post("/llm/phonely/session", json={}, headers=AUTH)).status_code == 503


async def test_session_translates_the_phonely_shape(
    client: AsyncClient,
    bind_phonely: Callable[[Handler], None],
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"callId": "call-1", "expiresAt": "later"})

    bind_phonely(handler)
    response = await client.post("/llm/phonely/session", json={"ignored": True}, headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"sessionId": "call-1", "expiresAt": "later"}
    assert captured[0].url.path == "/api/calls/session"


async def test_chat_buffers_the_stream_strips_metadata_and_records_timing(
    client: AsyncClient,
    postgresql: Any,
    bind_phonely: Callable[[Handler], None],
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=_sse({"role": "assistant"}, {"content": "x" * 4096}))

    bind_phonely(handler)
    response = await client.post(
        "/llm/phonely/chat",
        headers={**AUTH, "Accept-Encoding": "gzip"},
        json={
            "model": "call-1",
            "simulation_id": "sim-1",
            "messages": [
                {"role": "user", "content": "Hi", "timestamp": 1},
                {"role": "assistant", "content": "Earlier"},
                {"role": "user", "content": "Again"},
            ],
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "x" * 4096}
    assert "content-encoding" not in response.headers
    assert float(response.headers["X-Coval-Ttft-Ms"]) >= 0
    upstream = json.loads(captured[0].content)
    assert upstream["stream"] is True
    assert "timestamp" not in upstream["messages"][0]
    rows = await _turn_rows(postgresql)
    assert len(rows) == 1
    assert (rows[0]["simulation_id"], rows[0]["turn_index"]) == ("sim-1", 1)
    assert rows[0]["total_ms"] >= rows[0]["ttft_ms"] >= 0


async def test_tool_only_turn_has_string_content_and_openai_tool_calls(
    client: AsyncClient,
    bind_phonely: Callable[[Handler], None],
) -> None:
    tool_delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call-end",
                "type": "function",
                "function": {"name": "endCall", "arguments": "{}"},
            }
        ]
    }
    bind_phonely(lambda _request: httpx.Response(200, content=_sse(tool_delta)))
    response = await client.post(
        "/llm/phonely/chat",
        headers=AUTH,
        json={"model": "call-1", "messages": [{"role": "user", "content": "Bye"}]},
    )

    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == ""
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "endCall"
    assert choice["finish_reason"] == "tool_calls"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(401), 502),
        (httpx.Response(404), 502),
        (httpx.Response(500), 502),
        (httpx.Response(200, content=b'data: {"error":{"message":"failed"}}'), 502),
        (httpx.Response(200, content=b"data: [DONE]"), 502),
        (httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"partial"}}]}'), 502),
    ],
)
async def test_upstream_failures_are_proxy_failures(
    client: AsyncClient,
    bind_phonely: Callable[[Handler], None],
    response: httpx.Response,
    expected: int,
) -> None:
    bind_phonely(lambda _request: response)
    result = await client.post(
        "/llm/phonely/chat", headers=AUTH, json={"model": "call-1", "messages": []}
    )
    assert result.status_code == expected


async def test_streaming_requests_are_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/llm/phonely/chat",
        headers=AUTH,
        json={"model": "call-1", "messages": [], "stream": True},
    )
    assert response.status_code == 400


async def test_session_failures_are_proxy_failures(
    client: AsyncClient, bind_phonely: Callable[[Handler], None]
) -> None:
    bind_phonely(lambda _request: httpx.Response(500))
    response = await client.post("/llm/phonely/session", json={}, headers=AUTH)
    assert response.status_code == 502


class _StalledStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        await asyncio.sleep(3600)


async def test_a_stalled_turn_returns_504(
    client: AsyncClient,
    bind_phonely: Callable[[Handler], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("coval_bench.api.routers.llm_phonely._TURN_TIMEOUT_S", 0.2)
    bind_phonely(lambda _request: httpx.Response(200, stream=_StalledStream()))
    response = await client.post(
        "/llm/phonely/chat", headers=AUTH, json={"model": "call-1", "messages": []}
    )
    assert response.status_code == 504


async def test_failed_timing_insert_does_not_fail_the_turn(
    client: AsyncClient,
    bind_phonely: Callable[[Handler], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind_phonely(lambda _request: httpx.Response(200, content=_sse({"content": "Hi"})))
    monkeypatch.setattr(
        "coval_bench.api.routers.llm_phonely.insert_turn",
        AsyncMock(side_effect=RuntimeError("database unavailable")),
    )
    logger = MagicMock()
    monkeypatch.setattr("coval_bench.api.routers.llm_phonely.logger", logger)

    response = await client.post(
        "/llm/phonely/chat",
        headers=AUTH,
        json={"model": "call-1", "messages": [], "simulation_id": "sim-1"},
    )

    assert response.status_code == 200
    logger.error.assert_called_once()
    assert logger.error.call_args.args[0] == "llm_turn_not_recorded"


async def test_proxy_is_not_rate_limited(
    client: AsyncClient,
    bind_phonely: Callable[[Handler], None],
) -> None:
    bind_phonely(lambda _request: httpx.Response(200, content=_sse({"content": "Hi"})))
    responses = await asyncio.gather(
        *(
            client.post("/llm/phonely/chat", headers=AUTH, json={"model": "call-1", "messages": []})
            for _ in range(5)
        )
    )
    assert {response.status_code for response in responses} == {200}
