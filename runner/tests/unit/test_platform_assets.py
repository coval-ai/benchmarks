# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import coval_bench.platform_assets as platform_assets
from coval_bench.assets import SecretRef
from coval_bench.contracts import read_contract_file
from coval_bench.platform_assets import (
    AGENTS,
    TOOL_TIMEOUT_SECONDS,
    CovalClient,
    Plan,
    SyncError,
    VapiClient,
    apply,
    coval_agent_body,
    desired,
    drift,
    launch_body,
    patch_body,
    plan,
    platform_for,
    probe_vapi,
    register,
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
    monkeypatch.setenv("VAPI_DENTAL_DIAL_TARGET", "sip:appointment-dental@sip.vapi.ai")
    monkeypatch.setattr(platform_assets, "has_private_contract", lambda suite: True)


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


# --- dry run ---------------------------------------------------------------


def test_apply_dry_run_plans_but_never_patches(env: None) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(200, json=LIVE)

    with _client(handler) as client:
        result = apply(client, DENTAL, desired(DENTAL, BASE), dry_run=True)
    assert list(result.update) == ["model.tools"]
    assert result.prepared == []
    assert seen == ["GET"]


def test_platform_without_prepare_reports_nothing_prepared(env: None) -> None:
    assert platform_for(DENTAL).prepare is None
    with _client(lambda r: httpx.Response(200, json=LIVE)) as client:
        assert apply(client, DENTAL, desired(DENTAL, BASE)).prepared == []


def test_plan_summary_names_all_three_buckets() -> None:
    result = Plan(update={"a": (1, 2)}, unchanged=["b"], prepared=["secret:x"])
    assert result.summary() == "prepare=['secret:x'] update=['a'] unchanged=['b']"


# --- probe -----------------------------------------------------------------


def test_vapi_probe_sends_the_sdk_envelope_at_the_tool_url() -> None:
    (tool, *_) = render_vapi_tools(CONTRACT, BASE, SECRET)
    url, headers, body = probe_vapi(tool, {"phone": "6025550182"}, SECRET, "sim-1")
    assert url == "https://mock.example.com/mock/vapi"
    assert headers == {"X-Mock-Tools-Key": SECRET, "X-Coval-Simulation-Id": "sim-1"}
    (call,) = body["message"]["toolCallList"]
    assert call["id"] == "sim-1-1"
    assert call["function"] == {"name": "lookup_patient", "arguments": '{"phone": "6025550182"}'}


# --- coval side ------------------------------------------------------------


def test_coval_agent_body_dials_the_target_with_the_pinned_codec(env: None) -> None:
    body = coval_agent_body(DENTAL)
    assert body["model_type"] == "MODEL_TYPE_VOICE"
    assert body["phone_number"] == "sip:appointment-dental@sip.vapi.ai"
    assert body["metadata"] == {"audio_codec": "PCMU"}
    assert body["customer_agent_id"] == "vapi-dental"
    assert body["attributes"]["platform"] == "vapi"
    assert body["tags"] == ["orchestration", "dental", "vapi"]
    assert body["prompt"].startswith("BrightSmile Dental")


def test_dial_target_is_required_and_named_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VAPI_DENTAL_DIAL_TARGET", raising=False)
    with pytest.raises(RuntimeError, match="VAPI_DENTAL_DIAL_TARGET is unset"):
        coval_agent_body(DENTAL)


def _coval_client(state: dict[str, Any]) -> CovalClient:
    def handler(request: httpx.Request) -> httpx.Response:
        state.setdefault("calls", []).append((request.method, request.url.path))
        if request.url.path == "/v1/agents" and request.method == "GET":
            return httpx.Response(200, json={"agents": state["agents"], "next_page_token": ""})
        if request.url.path == "/v1/agents":
            created = {**json.loads(request.content), "id": "A" * 22}
            state["agents"].append(created)
            return httpx.Response(200, json={"agent": created})
        if request.url.path.startswith("/v1/agents/"):
            state["patched"] = json.loads(request.content)
            return httpx.Response(200, json={"agent": state["agents"][0]})
        if request.url.path == "/v1/runs":
            state["launched"] = json.loads(request.content)
            return httpx.Response(200, json={"run": {"run_id": "R" * 22, "status": "PENDING"}})
        return httpx.Response(404, text=request.url.path)

    return CovalClient(
        "coval-key", "https://api.coval.dev/v1", transport=httpx.MockTransport(handler)
    )


def test_register_creates_the_coval_agent_when_absent(env: None) -> None:
    state: dict[str, Any] = {"agents": []}
    with _coval_client(state) as client:
        agent_id, result = register(client, DENTAL)
    assert agent_id == "A" * 22
    assert set(result.update) == set(platform_assets.COVAL_MANAGED)
    assert state["agents"][0]["phone_number"] == "sip:appointment-dental@sip.vapi.ai"


def test_register_patches_only_the_drifted_fields(env: None) -> None:
    live = {**coval_agent_body(DENTAL), "id": "A", "phone_number": "sip:old@sip.vapi.ai"}
    state: dict[str, Any] = {"agents": [live]}
    with _coval_client(state) as client:
        agent_id, result = register(client, DENTAL)
    assert agent_id == "A"
    assert list(result.update) == ["phone_number"]
    assert state["patched"] == {"phone_number": "sip:appointment-dental@sip.vapi.ai"}


def test_register_matches_on_customer_agent_id_not_display_name(env: None) -> None:
    other = {**coval_agent_body(DENTAL), "id": "B", "customer_agent_id": "someone-else"}
    state: dict[str, Any] = {"agents": [other]}
    with _coval_client(state) as client:
        agent_id, _ = register(client, DENTAL)
    assert agent_id == "A" * 22
    assert len(state["agents"]) == 2


def test_register_dry_run_writes_nothing(env: None) -> None:
    state: dict[str, Any] = {"agents": []}
    with _coval_client(state) as client:
        agent_id, result = register(client, DENTAL, dry_run=True)
    assert agent_id == ""
    assert len(result.update) == len(platform_assets.COVAL_MANAGED) == 6
    assert all(method == "GET" for method, _ in state["calls"])


def test_launch_body_is_reproducible_and_tagged_with_the_arm(env: None) -> None:
    body = launch_body("A" * 22, DENTAL, "P" * 22, "T" * 8, ("M" * 22,), 3, 42)
    assert body["agent_id"] == "A" * 22
    assert body["options"] == {
        "iteration_count": 1,
        "concurrency": 1,
        "sub_sample_size": 3,
        "sub_sample_seed": 42,
    }
    assert body["metric_ids"] == ["M" * 22]
    assert body["metadata"]["customer_metadata"]["arm"] == "vapi-dental"
    assert body["metadata"]["display_name"].startswith("vapi-dental e2e ")


def test_launch_run_unwraps_the_run(env: None) -> None:
    state: dict[str, Any] = {"agents": []}
    with _coval_client(state) as client:
        run = client.launch_run(launch_body("A" * 22, DENTAL, "P" * 22, "T" * 8, (), 1, 1))
    assert run == {"run_id": "R" * 22, "status": "PENDING"}
    assert state["launched"]["options"]["sub_sample_size"] == 1


def test_drift_lists_what_apply_would_still_change_without_writing(env: None) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(200, json=LIVE)

    with _client(handler) as client:
        assert drift(client, DENTAL, BASE) == ["model.tools"]
    assert seen == ["GET"]


def test_drift_is_empty_once_the_platform_matches(env: None) -> None:
    synced = {**LIVE, "model": {**LIVE["model"], "tools": desired(DENTAL, BASE)["model.tools"]}}
    with _client(lambda r: httpx.Response(200, json=synced)) as client:
        assert drift(client, DENTAL, BASE) == []


def test_register_reconciles_display_name_and_tags_too(env: None) -> None:
    live = {**coval_agent_body(DENTAL), "id": "A", "display_name": "old name", "tags": []}
    state: dict[str, Any] = {"agents": [live]}
    with _coval_client(state) as client:
        _, result = register(client, DENTAL)
    assert sorted(result.update) == ["display_name", "tags"]
    assert state["patched"] == {
        "display_name": "vapi-dental",
        "tags": ["orchestration", "dental", "vapi"],
    }


def test_register_refuses_a_non_voice_agent_with_our_customer_id(env: None) -> None:
    live = {**coval_agent_body(DENTAL), "id": "A", "model_type": "MODEL_TYPE_SMS"}
    with (
        _coval_client({"agents": [live]}) as client,
        pytest.raises(SyncError, match="MODEL_TYPE_SMS"),
    ):
        register(client, DENTAL)


def test_contract_hash_refuses_to_label_without_the_fixtures(
    env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform_assets, "has_private_contract", lambda suite: False)
    with pytest.raises(SyncError, match="seeded world"):
        coval_agent_body(DENTAL)
    with pytest.raises(SyncError, match="seeded world"):
        launch_body("A" * 22, DENTAL, "P" * 22, "T" * 8, (), 1, 1)
