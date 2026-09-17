import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

pytest.importorskip("livekit.agents")

from coval_bench.livekit_agent.contract import (
    endpointing_delay,
    keyterms,
    load_contract,
    prompt_sha256,
    read_tool_definitions,
)
from coval_bench.livekit_agent.correlation import from_attributes
from coval_bench.livekit_agent.tools import MockToolsClient, build_tools
from coval_bench.mocktools.codecs import Correlation
from coval_bench.platform_assets import render_livekit_tools

CORRELATION = Correlation("sim-123", "+15550100000", source="sip_header")


def test_contract_reads_the_shared_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEM_PROMPT", "You are the dental receptionist.")
    monkeypatch.delenv("FIRST_MESSAGE", raising=False)
    contract = load_contract("dental")
    assert contract.stack.stt.model == "nova-3"
    assert contract.stack.llm.temperature == 0
    assert [tool["name"] for tool in contract.tools] == [
        tool["name"] for tool in read_tool_definitions("dental")
    ]
    assert contract.system_prompt == "You are the dental receptionist."
    assert len(contract.digest) == 64


def test_prompt_can_arrive_base64_encoded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYSTEM_PROMPT", raising=False)
    monkeypatch.setenv("SYSTEM_PROMPT_B64", "WW91IGFyZSBsaW5lIG9uZS4KTGluZSB0d28uCg==")
    assert load_contract("dental").system_prompt == "You are line one.\nLine two."


def test_prompt_digest_tracks_the_override_while_the_public_digest_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIRST_MESSAGE", raising=False)
    monkeypatch.setenv("SYSTEM_PROMPT", "prompt A")
    first = load_contract("dental")
    monkeypatch.setenv("SYSTEM_PROMPT", "prompt B")
    second = load_contract("dental")
    assert first.digest == second.digest
    assert first.prompt_digest != second.prompt_digest
    assert first.prompt_digest == prompt_sha256("prompt A", first.first_message)


def test_missing_prompt_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYSTEM_PROMPT", raising=False)
    with pytest.raises(ValueError, match="system prompt"):
        load_contract("dental")


def test_endpointing_delay_lands_on_the_pinned_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """800 ms pin minus the 300 ms Silero silence window: the headline turn-taking parity number."""
    monkeypatch.setenv("SYSTEM_PROMPT", "p")
    contract = load_contract("dental")
    assert contract.stack.turn_taking.end_of_turn_target_ms == 800
    assert endpointing_delay(contract) == pytest.approx(0.5)
    short = replace(
        contract,
        stack=contract.stack.model_copy(
            update={
                "turn_taking": contract.stack.turn_taking.model_copy(
                    update={"end_of_turn_target_ms": 200}
                )
            }
        ),
    )
    assert endpointing_delay(short) == 0.0


def test_keyterms_are_required_and_split_on_commas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPGRAM_KEYTERMS", raising=False)
    with pytest.raises(ValueError, match="DEEPGRAM_KEYTERMS"):
        keyterms()
    monkeypatch.setenv("DEEPGRAM_KEYTERMS", " Invisalign, amoxicillin ,, ")
    assert keyterms() == ["Invisalign", "amoxicillin"]


def test_correlation_prefers_the_coval_header_then_the_sip_number() -> None:
    both = from_attributes({"sip.h.X-Coval-Simulation-Id": "sim-1", "sip.phoneNumber": "+1555"})
    assert both == Correlation("sim-1", "+1555", source="sip_header")
    assert from_attributes({"sip.phoneNumber": "+1555"}) == Correlation(
        None, "+1555", source="sip_phone"
    )
    assert from_attributes({}) == Correlation()


async def test_client_speaks_the_livekit_codec() -> None:
    seen: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["headers"] = {
            k: request.headers[k]
            for k in ("x-mock-tools-key", "x-coval-simulation-id", "x-coval-caller-number")
        }
        return httpx.Response(200, json={"patient_id": "p1"})

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = MockToolsClient("https://api.example/", "k", transport)
    reply = await client.call("lookup_patient", {"phone": "5550100000"}, CORRELATION)
    assert json.loads(reply) == {"patient_id": "p1"}
    assert seen == {
        "url": "https://api.example/mock/livekit/lookup_patient",
        "body": {"phone": "5550100000"},
        "headers": {
            "x-mock-tools-key": "k",
            "x-coval-simulation-id": "sim-123",
            "x-coval-caller-number": "+15550100000",
        },
    }


async def test_client_reports_failures_as_tool_output() -> None:
    failing = httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(503)))
    client = MockToolsClient("https://api.example", "k", failing)
    assert json.loads(await client.call("check_availability", {}, CORRELATION)) == {
        "error": "tool endpoint returned 503"
    }


async def test_tools_keep_the_contract_schema_and_route_by_name() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def poster(tool: str, args: dict[str, Any], _c: Correlation) -> str:
        calls.append((tool, args))
        return "{}"

    definitions = read_tool_definitions("dental")
    tools = build_tools(definitions, poster, lambda: CORRELATION)
    infos = [getattr(tool, "info") for tool in tools]  # noqa: B009
    assert [info.name for info in infos] == [d["name"] for d in definitions]
    assert infos[1].raw_schema["parameters"] == definitions[1]["parameters"]
    assert infos[1].raw_schema["description"] == definitions[1]["description"]
    await tools[1]({"date": "2026-10-01", "appointment_type": "cleaning"})  # type: ignore[operator]
    assert calls == [("check_availability", {"date": "2026-10-01", "appointment_type": "cleaning"})]


async def test_tools_resolve_the_correlation_at_call_time() -> None:
    seen: list[Correlation] = []
    holder = {"value": Correlation()}

    async def poster(_t: str, _a: dict[str, Any], correlation: Correlation) -> str:
        seen.append(correlation)
        return "{}"

    tools = build_tools(read_tool_definitions("dental")[:1], poster, lambda: holder["value"])
    await tools[0]({})  # type: ignore[operator]
    holder["value"] = CORRELATION
    await tools[0]({})  # type: ignore[operator]
    assert seen == [Correlation(), CORRELATION]


def test_livekit_renderer_is_the_contract_verbatim() -> None:
    definitions = read_tool_definitions("dental")
    rendered = render_livekit_tools(definitions, "https://ignored", "ignored")
    assert rendered == [
        {"name": d["name"], "description": d["description"], "parameters": d["parameters"]}
        for d in definitions
    ]
