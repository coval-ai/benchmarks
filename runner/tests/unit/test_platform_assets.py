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
    register,
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
    with pytest.raises(KeyError, match="known: telnyx-dental, vapi-dental"):
        spec_for("retell-dental")
    assert {s.key for s in AGENTS} == {"vapi-dental", "telnyx-dental"}


# --- render ----------------------------------------------------------------


def test_desired_resolves_the_secret_and_targets_model_tools(env: None) -> None:
    wanted = desired(DENTAL, BASE)
    assert list(wanted) == ["model.tools"]
    assert wanted["model.tools"][0]["server"]["headers"]["X-Mock-Tools-Key"] == SECRET


def test_desired_refuses_an_unknown_platform(env: None) -> None:
    stranger = DENTAL.model_copy(update={"platform": "retell"})
    with pytest.raises(SyncError, match="unknown platform 'retell'; known: telnyx, vapi"):
        desired(stranger, BASE)


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


# --- every platform, one case each ------------------------------------------

from dataclasses import dataclass  # noqa: E402

from coval_bench.mocktools.codecs import (  # noqa: E402
    PRESET_CALLER,
    PRESET_SIMULATION,
    Correlation,
    codec_for,
)
from coval_bench.platform_assets import (  # noqa: E402
    TELNYX_SECRETS,
    TelnyxClient,
    prepare_telnyx,
    sip_subdomain,
)

TELNYX_LIVE: dict[str, Any] = {
    "id": "assistant-1",
    "name": "Telnyx Dental Assistant",
    "model": "openai/gpt-4.1",
    "llm_api_key_ref": "",
    "transcription": {"model": "deepgram/nova-3-medical", "language": "en", "settings": {}},
    "voice_settings": {"voice": "Telnyx.Ultra.abc", "api_key_ref": None, "expressive_mode": True},
    "telephony_settings": {
        "default_texml_app_id": "texml-1",
        "recording_settings": {"enabled": True, "channels": "dual"},
    },
    "interruption_settings": {"start_speaking_plan": {"wait_seconds": 0.1}},
    "tools": [{"type": "hangup", "shared": True, "tool_id": "tool-1"}],
}


@dataclass(frozen=True)
class Case:
    """What a platform row must satisfy, phrased without naming its envelope."""

    key: str
    env: dict[str, str]
    live: dict[str, Any]
    agent_path: str
    update_method: str
    tool_name: Any
    tool_url: Any
    secret_in_config: bool
    expected_drift: frozenset[str]
    pins: dict[str, Any]
    wraps_in_data: bool
    echo: Any = None


CASES = [
    Case(
        key="vapi-dental",
        env={"VAPI_DENTAL_ASSISTANT_ID": "asst_1", "VAPI_API_KEY": "vapi-key"},
        live=LIVE,
        agent_path="/assistant/asst_1",
        update_method="PATCH",
        tool_name=lambda t: t["function"]["name"],
        tool_url=lambda t: t["server"]["url"],
        secret_in_config=True,
        expected_drift=frozenset({"model.tools"}),
        pins={},
        wraps_in_data=False,
    ),
    Case(
        key="telnyx-dental",
        env={
            "TELNYX_DENTAL_ASSISTANT_ID": "assistant-1",
            "TELNYX_API_KEY": "telnyx-key",
            "TELNYX_DENTAL_DIAL_TARGET": "sip:dental@coval-bench-dental.sip.telnyx.com",
            "OPENAI_API_KEY": "openai-key",
            "ELEVENLABS_API_KEY": "eleven-key",
        },
        live=TELNYX_LIVE,
        agent_path="/v2/ai/assistants/assistant-1",
        update_method="POST",
        tool_name=lambda t: t["webhook"]["name"],
        tool_url=lambda t: t["webhook"]["url"],
        secret_in_config=False,
        expected_drift=frozenset(
            {
                "tools",
                "llm_api_key_ref",
                "transcription.model",
                "voice_settings.voice",
                "voice_settings.api_key_ref",
                "voice_settings.expressive_mode",
                "tool_ids",
                "telephony_settings.recording_settings.enabled",
                "interruption_settings.start_speaking_plan.wait_seconds",
            }
        ),
        pins={
            "model": "openai/gpt-4.1",
            "llm_api_key_ref": "coval-bench-openai",
            "transcription.model": "deepgram/nova-3",
            "voice_settings.voice": "ElevenLabs.eleven_flash_v2.EXAVITQu4vr4xnSDxMaL",
            "voice_settings.api_key_ref": "coval-bench-elevenlabs",
            "voice_settings.expressive_mode": False,
            "tool_ids": [],
            "telephony_settings.recording_settings.enabled": False,
            "interruption_settings.start_speaking_plan.wait_seconds": 0.8,
        },
        wraps_in_data=True,
        echo=lambda patched: {
            **patched,
            "tool_ids": None,
            "tools": [
                {**tool, "tool_id": f"tool-{i}", "shared": False, "timeout_ms": 5000}
                for i, tool in enumerate(patched.get("tools", []))
            ],
        },
    ),
]


@pytest.fixture(params=CASES, ids=lambda c: c.key)
def case(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Case:
    chosen: Case = request.param
    for name, value in chosen.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MOCK_TOOLS_SECRET", SECRET)
    monkeypatch.setattr(platform_assets, "has_private_contract", lambda suite: True)
    return chosen


CLIENTS: dict[str, Any] = {"vapi": VapiClient, "telnyx": TelnyxClient}


def _platform_client(case: Case, state: dict[str, Any]) -> Any:  # noqa: ANN401
    """A fake vendor API: serves the live agent, records the patch, and answers Telnyx's extras."""
    spec = spec_for(case.key)
    platform = platform_for(spec)

    def agent(payload: dict[str, Any]) -> dict[str, Any]:
        return {"data": payload} if case.wraps_in_data else payload

    def handler(request: httpx.Request) -> httpx.Response:
        state.setdefault("calls", []).append((request.method, request.url.path))
        if request.url.path == case.agent_path and request.method == "GET":
            return httpx.Response(200, json=agent(state.get("live", case.live)))
        if request.url.path == case.agent_path:
            state["patched"] = json.loads(request.content)
            echoed = case.echo(state["patched"]) if case.echo else state["patched"]
            state["live"] = {**case.live, **echoed}
            return httpx.Response(200, json=agent(state["live"]))
        if request.url.path == "/v2/integration_secrets" and request.method == "GET":
            return httpx.Response(200, json={"data": [{"identifier": i} for i in state["secrets"]]})
        if request.url.path == "/v2/integration_secrets":
            state["secrets"].append(json.loads(request.content)["identifier"])
            return httpx.Response(201, json={"data": {"id": "s", "identifier": "x"}})
        if request.url.path == "/v2/texml_applications/texml-1" and request.method == "GET":
            inbound = {
                "sip_subdomain": state["sub"],
                "sip_subdomain_receive_settings": state["recv"],
            }
            return httpx.Response(200, json={"data": {"inbound": inbound}})
        if request.url.path == "/v2/texml_applications/texml-1":
            inbound = json.loads(request.content)["inbound"]
            state["sub"], state["recv"] = (
                inbound["sip_subdomain"],
                inbound["sip_subdomain_receive_settings"],
            )
            return httpx.Response(200, json={"data": {}})
        return httpx.Response(404, text=request.url.path)

    state.setdefault("secrets", list(TELNYX_SECRETS))
    state.setdefault("sub", "coval-bench-dental")
    state.setdefault("recv", "from_anyone")
    return CLIENTS[spec.platform]("key", platform.api_base, transport=httpx.MockTransport(handler))


def test_every_platform_renders_one_tool_per_contract_entry_in_order(case: Case) -> None:
    spec = spec_for(case.key)
    tools = desired(spec, BASE)[platform_for(spec).tools_path]
    assert [case.tool_name(t) for t in tools] == [d["name"] for d in CONTRACT]
    assert len(tools) == 5


def test_every_platform_targets_the_route_its_codec_expects(case: Case) -> None:
    spec = spec_for(case.key)
    codec = codec_for(spec.platform)
    for tool, definition in zip(
        desired(spec, BASE)[platform_for(spec).tools_path], CONTRACT, strict=True
    ):
        expected = codec.encode_request(definition["name"], {}, Correlation()).path
        assert case.tool_url(tool) == f"https://mock.example.com{expected}"


def test_every_platform_carries_the_contract_parameters_verbatim(case: Case) -> None:
    spec = spec_for(case.key)
    tools = desired(spec, BASE)[platform_for(spec).tools_path]
    for tool, definition in zip(tools, CONTRACT, strict=True):
        blob = json.dumps(tool, sort_keys=True)
        assert json.dumps(definition["parameters"], sort_keys=True) in blob
        assert definition["description"] in blob


def test_the_secret_is_a_value_or_a_reference_as_the_vendor_requires(case: Case) -> None:
    spec = spec_for(case.key)
    blob = json.dumps(desired(spec, BASE))
    assert (SECRET in blob) is case.secret_in_config


def test_every_platform_pins_what_its_row_declares(case: Case) -> None:
    wanted = desired(spec_for(case.key), BASE)
    for path, value in case.pins.items():
        assert wanted[path] == value


def test_plan_flags_exactly_the_drift_on_the_live_agent(case: Case) -> None:
    spec = spec_for(case.key)
    result = plan(case.live, desired(spec, BASE), platform_for(spec).canon)
    assert set(result.update) == set(case.expected_drift)


def test_dry_run_reports_without_writing_then_apply_converges(case: Case) -> None:
    spec = spec_for(case.key)
    wanted = desired(spec, BASE)
    state: dict[str, Any] = {}
    with _platform_client(case, state) as client:
        preview = apply(client, spec, wanted, dry_run=True)
        assert set(preview.update) == set(case.expected_drift)
        assert all(method == "GET" for method, _ in state["calls"])
        applied = apply(client, spec, wanted)
        assert (case.update_method, case.agent_path) in state["calls"]
        assert set(applied.update) == set(case.expected_drift)
        assert drift(client, spec, BASE) == []


def test_smoke_request_round_trips_through_the_codec(case: Case) -> None:
    codec = codec_for(spec_for(case.key).platform)
    request = codec.encode_request(
        "lookup_patient", {"phone": "6025550182"}, Correlation("sim-1", "+15550100000")
    )
    tool = "lookup_patient" if codec.tool_in_path else None
    (call,) = codec.decode(request.body, request.headers, tool)
    assert (call.tool, call.args) == ("lookup_patient", {"phone": "6025550182"})
    found = codec.correlate(request.body, request.headers)
    assert (found.simulation_id, found.caller_number) == ("sim-1", "+15550100000")


# --- telnyx only: prerequisites Vapi does not have -------------------------


@pytest.fixture
def telnyx_env(monkeypatch: pytest.MonkeyPatch) -> Case:
    chosen = next(c for c in CASES if c.key == "telnyx-dental")
    for name, value in chosen.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MOCK_TOOLS_SECRET", SECRET)
    return chosen


def test_telnyx_correlation_rides_in_preset_fields_the_model_cannot_see(telnyx_env: Case) -> None:
    (tool, *_) = desired(spec_for("telnyx-dental"), BASE)["tools"]
    assert tool["webhook"]["preset_body_fields"] == {
        PRESET_SIMULATION: "{{coval_simulation_id}}",
        PRESET_CALLER: "{{telnyx_end_user_target}}",
    }
    assert tool["webhook"]["headers"] == [
        {
            "name": "X-Mock-Tools-Key",
            "value": "{{#integration_secret}}coval-bench-mock{{/integration_secret}}",
        }
    ]


@pytest.mark.parametrize(
    "target",
    [
        "sip:dental@coval-bench-dental.sip.telnyx.com",
        "SIP:x@Coval-Bench-Dental.sip.telnyx.com:5060",
    ],
)
def test_sip_subdomain_is_read_off_the_dial_target(target: str) -> None:
    assert sip_subdomain(target) == "coval-bench-dental"


@pytest.mark.parametrize(
    "target", ["sip:dental@sip.vapi.ai", "+14045550710", "coval.sip.telnyx.com"]
)
def test_sip_subdomain_refuses_anything_else(target: str) -> None:
    with pytest.raises(SyncError):
        sip_subdomain(target)


def test_telnyx_prepare_lists_missing_secrets_and_subdomain_then_creates_only_those(
    telnyx_env: Case,
) -> None:
    spec = spec_for("telnyx-dental")
    state: dict[str, Any] = {"secrets": ["coval", "coval-bench-openai"], "sub": None}
    with _platform_client(telnyx_env, state) as client:
        pending = prepare_telnyx(client, spec, True)
        assert pending == [
            "integration_secret:coval-bench-elevenlabs",
            "integration_secret:coval-bench-mock",
            "sip_subdomain:texml-1=coval-bench-dental:from_anyone",
        ]
        assert state["secrets"] == ["coval", "coval-bench-openai"] and state["sub"] is None
        assert prepare_telnyx(client, spec, False) == pending
        assert prepare_telnyx(client, spec, False) == []
    assert set(state["secrets"]) == {"coval", *TELNYX_SECRETS}
    assert state["sub"] == "coval-bench-dental"


def test_telnyx_prepare_refuses_a_foreign_client(telnyx_env: Case) -> None:
    with pytest.raises(SyncError), _client(lambda r: httpx.Response(200, json={})) as vapi:
        prepare_telnyx(vapi, spec_for("telnyx-dental"), True)


def test_telnyx_client_updates_with_post_as_the_reference_documents() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(200, json={"data": {}})

    with TelnyxClient(
        "k", "https://api.telnyx.com/v2", transport=httpx.MockTransport(handler)
    ) as c:
        c.update_agent("assistant-1", {})
    assert seen == ["POST"]


def test_telnyx_prepare_reopens_a_subdomain_locked_to_own_connections(telnyx_env: Case) -> None:
    spec = spec_for("telnyx-dental")
    state: dict[str, Any] = {"sub": "coval-bench-dental", "recv": "only_my_connections"}
    with _platform_client(telnyx_env, state) as client:
        assert prepare_telnyx(client, spec, True) == [
            "sip_subdomain:texml-1=coval-bench-dental:from_anyone"
        ]
        prepare_telnyx(client, spec, False)
        assert prepare_telnyx(client, spec, False) == []
    assert state["recv"] == "from_anyone"


def test_telnyx_canon_ignores_what_the_server_adds_to_tools(telnyx_env: Case) -> None:
    from coval_bench.platform_assets import TELNYX_CANON

    (tool, *_) = desired(spec_for("telnyx-dental"), BASE)["tools"]
    echoed = [{**tool, "tool_id": "tool-1", "shared": False, "timeout_ms": 5000}]
    assert TELNYX_CANON["tools"](echoed) == [tool]
    assert TELNYX_CANON["tool_ids"](None) == []
    assert TELNYX_CANON["tool_ids"](["t"]) == ["t"]
