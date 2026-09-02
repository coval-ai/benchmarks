# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from coval_bench.assets import SecretRef
from coval_bench.contracts import read_contract_file
from coval_bench.platform_assets import (
    AGENTS,
    TOOL_TIMEOUT_SECONDS,
    SyncError,
    VapiClient,
    apply,
    desired,
    patch_body,
    plan,
    platform_for,
    render_vapi_tools,
    spec_for,
)
from coval_bench.variants.platforms import redact

BASE = "https://mock.example.com/"
SECRET = "s3cr3t-value"  # noqa: S105
CONTRACT = json.loads(read_contract_file("dental", "tool-definitions.json"))
DENTAL = spec_for("vapi-dental")

LIVE: dict[str, Any] = {
    "id": "asst_1",
    "orgId": "org_1",
    "name": "Vapi Dental",
    "model": {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0,
        "messages": [{"role": "system", "content": "You are the front desk."}],
        "tools": [],
    },
    "voice": {"provider": "vapi", "voiceId": "Elliot"},
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAPI_DENTAL_ASSISTANT_ID", "asst_1")
    monkeypatch.setenv("VAPI_API_KEY", "vapi-key")
    monkeypatch.setenv("MOCK_TOOLS_SECRET", SECRET)


def _client(handler: Any) -> VapiClient:  # noqa: ANN401
    return VapiClient("vapi-key", "https://api.vapi.ai", transport=httpx.MockTransport(handler))


# --- SecretRef -------------------------------------------------------------


def test_secret_ref_names_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NOPE_KEY is unset. It holds the thing"):
        SecretRef(name="NOPE_KEY", purpose="the thing").resolve()


def test_spec_for_names_the_known_set() -> None:
    with pytest.raises(KeyError, match="known: vapi-dental"):
        spec_for("telnyx-dental")
    assert {s.key for s in AGENTS} == {"vapi-dental"}


# --- render ----------------------------------------------------------------


def test_render_emits_one_function_tool_per_contract_entry() -> None:
    tools = render_vapi_tools(CONTRACT, BASE, SECRET)
    assert len(tools) == len(CONTRACT) == 5
    assert [t["function"]["name"] for t in tools] == [d["name"] for d in CONTRACT]
    assert all(t["type"] == "function" and t["async"] is False for t in tools)


def test_render_copies_parameters_verbatim() -> None:
    tools = render_vapi_tools(CONTRACT, BASE, SECRET)
    for tool, definition in zip(tools, CONTRACT, strict=True):
        assert tool["function"]["parameters"] == definition["parameters"]
        assert tool["function"]["description"] == definition["description"]


def test_render_points_every_tool_at_the_vapi_codec_with_both_headers() -> None:
    (tool, *_) = render_vapi_tools(CONTRACT, BASE, SECRET)
    assert tool["server"]["url"] == "https://mock.example.com/mock/vapi"
    assert tool["server"]["timeoutSeconds"] == TOOL_TIMEOUT_SECONDS
    assert tool["server"]["headers"] == {
        "X-Mock-Tools-Key": SECRET,
        "X-Coval-Simulation-Id": "{{coval-simulation-id}}",
    }


def test_render_never_speaks_filler_messages() -> None:
    assert all("messages" not in t for t in render_vapi_tools(CONTRACT, BASE, SECRET))


def test_render_is_pure() -> None:
    assert render_vapi_tools(CONTRACT, BASE, SECRET) == render_vapi_tools(CONTRACT, BASE, SECRET)


def test_desired_resolves_the_secret_and_targets_model_tools(env: None) -> None:
    wanted = desired(DENTAL, BASE)
    assert list(wanted) == ["model.tools"]
    assert wanted["model.tools"][0]["server"]["headers"]["X-Mock-Tools-Key"] == SECRET


def test_desired_refuses_an_unknown_platform(env: None) -> None:
    stranger = DENTAL.model_copy(update={"platform": "telnyx"})
    with pytest.raises(SyncError, match="unknown platform 'telnyx'; known: vapi"):
        desired(stranger, BASE)


def test_the_platform_table_supplies_render_and_client_together() -> None:
    platform = platform_for(DENTAL)
    assert platform.name == "vapi"
    assert platform.api_base == "https://api.vapi.ai"
    assert platform.client is VapiClient


def test_apply_accepts_any_client_with_the_two_methods(env: None) -> None:
    class Fake:
        def __init__(self) -> None:
            self.patched: dict[str, Any] | None = None

        def __enter__(self) -> Fake:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def get_agent(self, agent_id: str) -> dict[str, Any]:
            return LIVE

        def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
            self.patched = body
            return body

    fake = Fake()
    apply(fake, DENTAL, desired(DENTAL, BASE))
    assert fake.patched is not None and len(fake.patched["model"]["tools"]) == 5


# --- plan ------------------------------------------------------------------


def test_plan_reports_tools_as_the_one_update(env: None) -> None:
    result = plan(LIVE, desired(DENTAL, BASE))
    assert list(result.update) == ["model.tools"]
    assert result.unchanged == []
    before, after = result.update["model.tools"]
    assert before == []
    assert len(after) == 5


def test_plan_is_unchanged_when_live_already_matches(env: None) -> None:
    wanted = desired(DENTAL, BASE)
    live = {**LIVE, "model": {**LIVE["model"], "tools": wanted["model.tools"]}}
    result = plan(live, wanted)
    assert result.update == {}
    assert result.unchanged == ["model.tools"]


def test_plan_output_is_safe_to_print(env: None) -> None:
    (_, after) = plan(LIVE, desired(DENTAL, BASE)).update["model.tools"]
    found: list[str] = []
    printed = json.dumps(redact(after, found))
    assert SECRET not in printed
    assert "{{coval-simulation-id}}" in printed
    assert any(path.endswith("X-Mock-Tools-Key") for path in found)


# --- apply -----------------------------------------------------------------


def test_patch_body_carries_the_whole_live_model_with_tools_replaced(env: None) -> None:
    body = patch_body(LIVE, desired(DENTAL, BASE))
    assert list(body) == ["model"]
    assert body["model"]["messages"] == LIVE["model"]["messages"]
    assert body["model"]["provider"] == "openai"
    assert body["model"]["temperature"] == 0
    assert len(body["model"]["tools"]) == 5
    assert LIVE["model"]["tools"] == []


def test_apply_gets_then_patches_only_the_model(env: None) -> None:
    calls: list[tuple[str, str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, payload))
        return httpx.Response(200, json=LIVE)

    with _client(handler) as client:
        result = apply(client, DENTAL, desired(DENTAL, BASE))

    assert [(m, p) for m, p, _ in calls] == [
        ("GET", "/assistant/asst_1"),
        ("PATCH", "/assistant/asst_1"),
    ]
    patched = calls[1][2]
    assert set(patched) == {"model"}
    assert "id" not in patched and "orgId" not in patched
    assert patched["model"]["messages"] == LIVE["model"]["messages"]
    assert len(patched["model"]["tools"]) == 5
    assert list(result.update) == ["model.tools"]


def test_apply_skips_the_patch_when_nothing_changed(env: None) -> None:
    wanted = desired(DENTAL, BASE)
    live = {**LIVE, "model": {**LIVE["model"], "tools": wanted["model.tools"]}}
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=live)

    with _client(handler) as client:
        result = apply(client, DENTAL, wanted)
    assert methods == ["GET"]
    assert result.update == {}


def test_apply_surfaces_vendor_errors(env: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=LIVE)
        return httpx.Response(400, text="voiceId must be a string")

    with (
        _client(handler) as client,
        pytest.raises(SyncError, match="PATCH /assistant/asst_1 -> 400"),
    ):
        apply(client, DENTAL, desired(DENTAL, BASE))
