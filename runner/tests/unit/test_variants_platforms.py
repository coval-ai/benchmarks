# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading a Retell agent and the LLM it points at."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from coval_bench.variants.platforms import read_retell, retell_engine

Handler = Any

RETELL_AGENT: dict[str, Any] = {
    "agent_id": "agent_1",
    "response_engine": {"type": "retell-llm", "llm_id": "llm_1", "version": None},
    "voice_id": "11labs-Adrian",
}
RETELL_LLM: dict[str, Any] = {
    "llm_id": "llm_1",
    "general_prompt": "You are the front desk.",
    "begin_message": " Thanks for calling. ",
    "general_tools": [{"type": "custom", "name": "lookup_patient", "url": "https://m/x"}],
}


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(base_url="https://api.retellai.com", transport=httpx.MockTransport(handler))


def _routes(table: dict[str, dict[str, Any]], seen: list[tuple[str, str]]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, str(request.url.query, "ascii")))
        body = table.get(request.url.path)
        return httpx.Response(200 if body is not None else 404, json=body or {})

    return handler


def _get(client: httpx.Client) -> Any:  # noqa: ANN401
    return lambda path, params: client.get(path, params=params).json()


def test_read_retell_reads_the_agent_then_the_llm_it_points_at() -> None:
    seen: list[tuple[str, str]] = []
    table = {"/get-agent/agent_1": RETELL_AGENT, "/get-retell-llm/llm_1": RETELL_LLM}
    with _client(_routes(table, seen)) as client:
        agent, llm = read_retell(_get(client), "agent_1")
    assert seen == [("/get-agent/agent_1", ""), ("/get-retell-llm/llm_1", "")]
    assert (agent, llm) == (RETELL_AGENT, RETELL_LLM)


def test_read_retell_reads_the_llm_version_the_agent_pins() -> None:
    seen: list[tuple[str, str]] = []
    pinned = {**RETELL_AGENT, "response_engine": {**RETELL_AGENT["response_engine"], "version": 7}}
    table = {"/get-agent/agent_1": pinned, "/get-retell-llm/llm_1": RETELL_LLM}
    with _client(_routes(table, seen)) as client:
        read_retell(_get(client), "agent_1")
    assert seen[1] == ("/get-retell-llm/llm_1", "version=7")


@pytest.mark.parametrize(
    "engine",
    [
        {"type": "conversation-flow", "conversation_flow_id": "cf_1"},
        {"type": "custom-llm", "llm_websocket_url": "wss://x"},
        None,
    ],
)
def test_retell_refuses_agents_without_a_retell_llm(engine: dict[str, Any] | None) -> None:
    with pytest.raises(RuntimeError, match="only retell-llm carries the prompt and tools"):
        retell_engine({**RETELL_AGENT, "response_engine": engine})


def test_read_retell_refuses_a_multi_state_llm() -> None:
    llm = {**RETELL_LLM, "states": [{"name": "book", "state_prompt": "Book.", "tools": []}]}
    table = {"/get-agent/agent_1": RETELL_AGENT, "/get-retell-llm/llm_1": llm}
    with (
        _client(_routes(table, [])) as client,
        pytest.raises(RuntimeError, match="has 1 states; their prompts and tools stay live"),
    ):
        read_retell(_get(client), "agent_1")
