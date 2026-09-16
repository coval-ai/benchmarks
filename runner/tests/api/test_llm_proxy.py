# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The authenticated LLM proxy, exercised through the Phonely provider."""

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
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import RegisteredModel
from tests.api.conftest import LLM_PROXY_KEY, _make_db_url, add_models

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


def _llm_model(provider: str, *, collected: bool) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.LLM,
        provider=provider,
        model=f"{provider}-agent",
        collected=collected,
        published=False,
    )


@pytest.fixture(autouse=True)
def llm_roster(postgresql: Any) -> None:
    add_models(
        postgresql, _llm_model("phonely", collected=True), _llm_model("paused", collected=False)
    )


@pytest_asyncio.fixture
async def bind_phonely(app: FastAPI) -> AsyncIterator[Callable[[Handler], None]]:
    handlers: list[Handler] = []
    client = PhonelyClient(
        "upstream-key",
        "agent-1",
        "https://phonely.test",
        transport=httpx.MockTransport(lambda request: handlers[-1](request)),
    )
    app.state.llm_clients = {("phonely", "phonely-agent"): client}
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
    configured_clients = app.state.llm_clients
    assert isinstance(configured_clients["phonely", "phonely-agent"], PhonelyClient)
    assert (await client.post("/llm/phonely/session", json={})).status_code == 401
    assert (
        await client.post(
            "/llm/phonely/session", json={}, headers={"Authorization": "Bearer wrong"}
        )
    ).status_code == 401
    assert (await client.post("/v1/llm/phonely/session", json={}, headers=AUTH)).status_code == 404
    assert (await client.post("/llm/unknown/session", json={}, headers=AUTH)).status_code == 404
    assert (await client.post("/llm/paused/session", json={}, headers=AUTH)).status_code == 404

    original_settings = app.state.settings
    app.state.settings = original_settings.model_copy(update={"phonely_api_key": None})
    app.state.llm_clients = {}
    assert (await client.post("/llm/phonely/session", json={}, headers=AUTH)).status_code == 503
    lowercase = {"Authorization": f"bearer {LLM_PROXY_KEY}"}
    lowercase_response = await client.post("/llm/phonely/session", json={}, headers=lowercase)
    assert lowercase_response.status_code == 503
    app.state.settings = original_settings
    app.state.llm_clients = configured_clients
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
    monkeypatch.setattr("coval_bench.api.routers.llm_proxy._TURN_TIMEOUT_S", 0.2)
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
        "coval_bench.api.routers.llm_proxy.insert_turn",
        AsyncMock(side_effect=RuntimeError("database unavailable")),
    )
    logger = MagicMock()
    monkeypatch.setattr("coval_bench.api.routers.llm_proxy.logger", logger)

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


async def test_openai_variants_route_and_record_the_selected_model(
    client: AsyncClient, app: FastAPI, postgresql: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=_sse({"content": "Welcome to Ultra Bank"}))

    # Exercise the real factories and streaming parser with only transport stubbed.
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **_kw: httpx.MockTransport(handler))
    app.state.settings = app.state.settings.model_copy(update={"openai_api_key": SecretStr("test")})
    names = ["gpt-4.1", "gpt-5-minimal", "gpt-5-medium"]
    add_models(
        postgresql,
        *[
            RegisteredModel(
                benchmark=Benchmark.LLM,
                provider="openai",
                model=name,
                collected=True,
                published=False,
            )
            for name in names
        ],
    )
    previous = app.state.llm_clients
    app.state.llm_clients = {}
    try:
        for index, name in enumerate(names):
            # The legacy provider-only endpoint must keep serving GPT-4.1.
            query = "" if name == "gpt-4.1" else f"?benchmark_model={name}"
            session = await client.post(f"/llm/openai/session{query}", json={}, headers=AUTH)
            assert session.status_code == 200
            response = await client.post(
                f"/llm/openai/chat{query}",
                headers=AUTH,
                json={
                    "model": session.json()["sessionId"],
                    "messages": [{"role": "user", "content": "Hello"}],
                    "simulation_id": f"variant-{index}",
                },
            )
            assert response.status_code == 200, response.text
        assert [body["model"] for body in captured] == ["gpt-4.1", "gpt-5", "gpt-5"]
        assert [body.get("reasoning_effort") for body in captured] == [None, "minimal", "medium"]
        assert len({json.dumps(body["messages"]) for body in captured}) == 1
        rows = await _turn_rows(postgresql)
        assert [(row["simulation_id"], row["model"]) for row in rows] == [
            (f"variant-{i}", name) for i, name in enumerate(names)
        ]
        assert all(row["provider"] == "openai" and row["ttft_ms"] >= 0 for row in rows)
        assert set(app.state.llm_clients) == {("openai", name) for name in names}
        for name in ["unknown", "gpt-5"]:
            response = await client.post(
                f"/llm/openai/session?benchmark_model={name}", headers=AUTH, json={}
            )
            assert response.status_code == 404
        # Removing collection must take effect even while the client is cached.
        with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
            conn.execute(
                "UPDATE benchmarks_v2.models SET collected=false "
                "WHERE modality='LLM' AND provider='openai' AND model='gpt-5-medium'"
            )
        response = await client.post(
            "/llm/openai/session?benchmark_model=gpt-5-medium", headers=AUTH, json={}
        )
        assert response.status_code == 404
    finally:
        for upstream in app.state.llm_clients.values():
            await upstream.aclose()
        app.state.llm_clients = previous
