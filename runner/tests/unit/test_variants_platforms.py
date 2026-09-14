# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One fetch per platform row: where the prompt, greeting and tools are read from."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from coval_bench.variants.platforms import FETCHERS, fetch_platform, read_retell, retell_engine

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


def _client(platform: str, handler: Handler) -> httpx.Client:
    return httpx.Client(
        base_url=FETCHERS[platform].api_base, transport=httpx.MockTransport(handler)
    )


def _routes(table: dict[str, dict[str, Any]], seen: list[tuple[str, str]]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, str(request.url.query, "ascii")))
        body = table.get(request.url.path)
        return httpx.Response(200 if body is not None else 404, json=body or {})

    return handler


def test_retell_reads_the_agent_then_the_llm_it_points_at() -> None:
    seen: list[tuple[str, str]] = []
    table = {"/get-agent/agent_1": RETELL_AGENT, "/get-retell-llm/llm_1": RETELL_LLM}
    with _client("retell", _routes(table, seen)) as client:
        config = FETCHERS["retell"].fetch(client, "agent_1")
    assert seen == [("/get-agent/agent_1", ""), ("/get-retell-llm/llm_1", "")]
    assert set(config.raw) == {"agent", "llm"}
    assert config.system_prompt == "You are the front desk."
    assert config.first_message == "Thanks for calling."
    assert [t["name"] for t in config.tools] == ["lookup_patient"]


def test_retell_reads_the_llm_version_the_agent_pins() -> None:
    seen: list[tuple[str, str]] = []
    pinned = {**RETELL_AGENT, "response_engine": {**RETELL_AGENT["response_engine"], "version": 7}}
    table = {"/get-agent/agent_1": pinned, "/get-retell-llm/llm_1": RETELL_LLM}
    with _client("retell", _routes(table, seen)) as client:
        FETCHERS["retell"].fetch(client, "agent_1")
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


def test_vapi_reads_the_system_message_and_model_tools() -> None:
    live = {
        "id": "asst_1",
        "firstMessage": " Hello. ",
        "model": {
            "messages": [{"role": "system", "content": "Be brief."}],
            "tools": [{"type": "function", "function": {"name": "lookup_patient"}}],
        },
    }
    with _client("vapi", _routes({"/assistant/asst_1": live}, [])) as client:
        config = FETCHERS["vapi"].fetch(client, "asst_1")
    assert (config.system_prompt, config.first_message) == ("Be brief.", "Hello.")
    assert config.tools[0]["function"]["name"] == "lookup_patient"


def test_telnyx_unwraps_the_data_envelope() -> None:
    live = {"data": {"instructions": "Be brief.", "greeting": "Hi.", "tools": [{"type": "hangup"}]}}
    with _client("telnyx", _routes({"/v2/ai/assistants/assistant-1": live}, [])) as client:
        config = FETCHERS["telnyx"].fetch(client, "assistant-1")
    assert config.raw == live["data"]
    assert (config.system_prompt, config.first_message) == ("Be brief.", "Hi.")
    assert config.tools == [{"type": "hangup"}]


def test_fetch_platform_names_the_known_set() -> None:
    with pytest.raises(KeyError, match="known: retell, telnyx, vapi"):
        fetch_platform("synthflow", "x")


def test_fetch_platform_names_the_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RETELL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RETELL_API_KEY is not set"):
        fetch_platform("retell", "agent_1")


def test_every_fetcher_is_registered_under_its_own_name() -> None:
    assert all(name == fetcher.name for name, fetcher in FETCHERS.items())
    assert json.dumps(sorted(FETCHERS)) == '["retell", "telnyx", "vapi"]'


def test_read_retell_refuses_a_multi_state_llm() -> None:
    llm = {**RETELL_LLM, "states": [{"name": "book", "state_prompt": "Book.", "tools": []}]}
    table = {"/get-agent/agent_1": RETELL_AGENT, "/get-retell-llm/llm_1": llm}
    with (
        _client("retell", _routes(table, [])) as client,
        pytest.raises(RuntimeError, match="has 1 states; their prompts and tools stay live"),
    ):
        read_retell(lambda path, params: client.get(path, params=params).json(), "agent_1")
