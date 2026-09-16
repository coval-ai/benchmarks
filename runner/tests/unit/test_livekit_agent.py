import json
from typing import Any

import httpx
import pytest

pytest.importorskip("livekit.agents")

from coval_bench.livekit_agent.contract import load_contract, read_tool_definitions
from coval_bench.livekit_agent.correlation import from_attributes
from coval_bench.livekit_agent.tools import MockToolsClient, build_tools
from coval_bench.mocktools.codecs import Correlation

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


def test_missing_prompt_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYSTEM_PROMPT", raising=False)
    with pytest.raises(ValueError, match="system prompt"):
        load_contract("dental")


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
    tools = build_tools(definitions, poster, CORRELATION)
    infos = [getattr(tool, "info") for tool in tools]  # noqa: B009
    assert [info.name for info in infos] == [d["name"] for d in definitions]
    assert infos[1].raw_schema["parameters"] == definitions[1]["parameters"]
    assert infos[1].raw_schema["description"] == definitions[1]["description"]
    await tools[1]({"date": "2026-10-01", "appointment_type": "cleaning"})  # type: ignore[operator]
    assert calls == [("check_availability", {"date": "2026-10-01", "appointment_type": "cleaning"})]
