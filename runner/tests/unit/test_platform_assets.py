# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import click
import httpx
import pytest
from click.testing import CliRunner

import coval_bench.platform_assets as platform_assets
import coval_bench.scenarios as scenarios
from coval_bench.assets import SecretRef
from coval_bench.mocktools import dispatch
from coval_bench.mocktools.dispatch import vendor_tool_definitions
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
    mock_base_url_for,
    patch_body,
    plan,
    platform_for,
    register,
    register_dial_target,
    spec_for,
    vendor_drift,
)
from coval_bench.variants.platforms import redact

BASE = "https://mock.example.com/"
SECRET = "s3cr3t-value"  # noqa: S105
TOOLS = json.dumps(
    {
        "tools": [
            {
                "name": "lookup",
                "description": "Look up the detail the caller gave.",
                "parameters": {"value": "string"},
                "character_error_rate": {"value": {"card-1": "4111111111111111"}},
            }
        ],
        "responses": {"lookup": {"received": True}},
    }
).encode()
CONTRACT = vendor_tool_definitions(TOOLS)
PROMPT = "You look up a caller's detail. Begin with: What can I look up for you?"
VAPI = spec_for("vapi/pinned")
# The full Coval reconcile, for an agent the assets do not own.
UNMAPPED = VAPI.model_copy(update={"suite": "dental", "scenario": None})


def _published(suite: str, name: str) -> bytes:
    files = {"tools.json": TOOLS, "system-prompt.txt": f"{PROMPT}\n".encode()}
    if suite != "id_roulette" or name not in files:
        raise FileNotFoundError(name)
    return files[name]


LIVE: dict[str, Any] = {
    "id": "asst_1",
    "orgId": "org_1",
    "name": "Vapi Dental",
    "model": {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0,
        "messages": [{"role": "system", "content": PROMPT}],
        "tools": [],
    },
    "voice": {"provider": "vapi", "voiceId": "Elliot"},
    "firstMessageMode": "assistant-speaks-first-with-model-generated-message",
}


@pytest.fixture(autouse=True)
def published(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scenarios, "_FIXTURE_PROVIDERS", [_published])
    monkeypatch.setattr(dispatch, "read_private_file", _published)
    monkeypatch.setattr(scenarios, "read_private_file", _published)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAPI_ID_ROULETTE_ASSISTANT_ID", "asst_1")
    monkeypatch.setenv("VAPI_API_KEY", "vapi-key")
    monkeypatch.setenv("MOCK_TOOLS_SECRET", SECRET)
    monkeypatch.setenv("VAPI_ID_ROULETTE_DIAL_TARGET", "sip:example-dental@sip.vapi.ai")
    monkeypatch.setattr(platform_assets, "has_private_contract", lambda suite: True)


def _client(handler: Any) -> VapiClient:  # noqa: ANN401
    return VapiClient("vapi-key", "https://api.vapi.ai", transport=httpx.MockTransport(handler))


# --- SecretRef -------------------------------------------------------------


def test_secret_ref_names_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NOPE_KEY is unset. It holds the thing"):
        SecretRef(name="NOPE_KEY", purpose="the thing").resolve()


def test_spec_for_names_the_known_set() -> None:
    with pytest.raises(KeyError, match="known: livekit/pinned, retell/pinned, telnyx/pinned, vapi"):
        spec_for("synthflow/pinned")
    assert {s.key for s in AGENTS} == {
        "vapi/pinned",
        "telnyx/pinned",
        "retell/pinned",
        "livekit/pinned",
    }


# --- code-managed platforms --------------------------------------------------


def test_code_managed_platform_refuses_plan_and_apply(env: None) -> None:
    spec = spec_for("livekit/pinned")
    assert spec.api_key is None
    with pytest.raises(SyncError, match="code-managed"):
        platform_for(spec)
    with pytest.raises(SyncError, match="code-managed"):
        desired(spec, BASE)


def test_code_managed_platform_has_no_vendor_drift_to_block_launch(env: None) -> None:
    assert vendor_drift(spec_for("livekit/pinned"), BASE, None) == []


def test_code_managed_agent_still_registers_with_coval(
    env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = "sip:+15550100001@example.sip.livekit.cloud;transport=tcp"
    monkeypatch.setenv("LIVEKIT_ID_ROULETTE_DIAL_TARGET", target)
    body = coval_agent_body(
        spec_for("livekit/pinned").model_copy(update={"suite": "dental", "scenario": None})
    )
    assert body["phone_number"] == target
    assert body["attributes"]["platform"] == "livekit"
    assert body["tags"] == ["orchestration", "dental", "livekit"]


def test_mock_base_url_is_optional_only_where_a_code_managed_arm_never_reads_it() -> None:
    livekit = spec_for("livekit/pinned")
    assert mock_base_url_for(livekit, None, needed_for_code_managed=False) == ""
    assert mock_base_url_for(livekit, BASE, needed_for_code_managed=False) == BASE
    with pytest.raises(click.UsageError, match="livekit/pinned"):
        mock_base_url_for(livekit, None, needed_for_code_managed=True)
    with pytest.raises(click.UsageError, match="vapi/pinned"):
        mock_base_url_for(VAPI, None, needed_for_code_managed=False)


def test_launch_cli_demands_the_mock_url_for_vendor_arms_only(
    env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MOCK_TOOLS_BASE_URL", raising=False)
    monkeypatch.delenv("COVAL_API_KEY", raising=False)
    common = ["launch", "--persona-id", "p", "--test-set-id", "t"]
    vendor = CliRunner().invoke(
        platform_assets.platform_assets, [*common, "--agent", "vapi/pinned"]
    )
    assert vendor.exit_code == 2
    assert "--mock-base-url" in vendor.output
    code_managed = CliRunner().invoke(
        platform_assets.platform_assets, [*common, "--agent", "livekit/pinned"]
    )
    assert "--mock-base-url" not in code_managed.output
    assert isinstance(code_managed.exception, RuntimeError)
    assert "COVAL_API_KEY" in str(code_managed.exception)


# --- render ----------------------------------------------------------------


def test_desired_resolves_the_secret_and_targets_model_tools(env: None) -> None:
    wanted = desired(VAPI, BASE)
    assert list(wanted) == ["model.tools", "model.messages", "firstMessageMode"]
    assert wanted["model.tools"][0]["server"]["headers"]["X-Mock-Tools-Key"] == SECRET


def test_desired_refuses_an_unknown_platform(env: None) -> None:
    stranger = VAPI.model_copy(update={"platform": "synthflow"})
    with pytest.raises(
        SyncError, match="unknown platform 'synthflow'; known: retell, telnyx, vapi"
    ):
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
    apply(fake, VAPI, desired(VAPI, BASE))
    assert fake.patched is not None and len(fake.patched["model"]["tools"]) == len(CONTRACT)


# --- plan ------------------------------------------------------------------


def test_plan_reports_tools_as_the_one_update(env: None) -> None:
    result = plan(LIVE, desired(VAPI, BASE))
    assert list(result.update) == ["model.tools"]
    assert sorted(result.unchanged) == ["firstMessageMode", "model.messages"]
    before, after = result.update["model.tools"]
    assert before == []
    assert len(after) == len(CONTRACT)


def test_plan_is_unchanged_when_live_already_matches(env: None) -> None:
    wanted = desired(VAPI, BASE)
    live = {**LIVE, "model": {**LIVE["model"], "tools": wanted["model.tools"]}}
    result = plan(live, wanted)
    assert result.update == {}
    assert sorted(result.unchanged) == ["firstMessageMode", "model.messages", "model.tools"]


def test_plan_output_is_safe_to_print(env: None) -> None:
    (_, after) = plan(LIVE, desired(VAPI, BASE)).update["model.tools"]
    found: list[str] = []
    printed = json.dumps(redact(after, found))
    assert SECRET not in printed
    assert "{{coval-simulation-id}}" in printed
    assert any(path.endswith("X-Mock-Tools-Key") for path in found)


# --- apply -----------------------------------------------------------------


def test_patch_body_carries_the_whole_live_model_with_tools_replaced(env: None) -> None:
    body = patch_body(LIVE, desired(VAPI, BASE))
    assert list(body) == ["model", "firstMessageMode"]
    assert body["model"]["messages"] == LIVE["model"]["messages"]
    assert body["model"]["provider"] == "openai"
    assert body["model"]["temperature"] == 0
    assert len(body["model"]["tools"]) == len(CONTRACT)
    assert LIVE["model"]["tools"] == []


def test_apply_gets_then_patches_only_the_model(env: None) -> None:
    calls: list[tuple[str, str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, payload))
        return httpx.Response(200, json=LIVE)

    with _client(handler) as client:
        result = apply(client, VAPI, desired(VAPI, BASE))

    assert [(m, p) for m, p, _ in calls] == [
        ("GET", "/assistant/asst_1"),
        ("PATCH", "/assistant/asst_1"),
    ]
    patched = calls[1][2]
    assert set(patched) == {"model"}
    assert "id" not in patched and "orgId" not in patched
    assert patched["model"]["messages"] == LIVE["model"]["messages"]
    assert len(patched["model"]["tools"]) == len(CONTRACT)
    assert list(result.update) == ["model.tools"]


def test_apply_skips_the_patch_when_nothing_changed(env: None) -> None:
    wanted = desired(VAPI, BASE)
    live = {**LIVE, "model": {**LIVE["model"], "tools": wanted["model.tools"]}}
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=live)

    with _client(handler) as client:
        result = apply(client, VAPI, wanted)
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
        apply(client, VAPI, desired(VAPI, BASE))


# --- dry run ---------------------------------------------------------------


def test_apply_dry_run_plans_but_never_patches(env: None) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(200, json=LIVE)

    with _client(handler) as client:
        result = apply(client, VAPI, desired(VAPI, BASE), dry_run=True)
    assert list(result.update) == ["model.tools"]
    assert result.prepared == []
    assert seen == ["GET"]


def test_platform_without_prepare_reports_nothing_prepared(env: None) -> None:
    assert platform_for(VAPI).prepare is None
    with _client(lambda r: httpx.Response(200, json=LIVE)) as client:
        assert apply(client, VAPI, desired(VAPI, BASE)).prepared == []


def test_plan_summary_names_all_three_buckets() -> None:
    result = Plan(update={"a": (1, 2)}, unchanged=["b"], prepared=["secret:x"])
    assert result.summary() == "prepare=['secret:[REDACTED]'] update=['a'] unchanged=['b']"


# --- probe -----------------------------------------------------------------


# --- coval side ------------------------------------------------------------


def test_coval_agent_body_dials_the_target_with_the_pinned_codec(env: None) -> None:
    body = coval_agent_body(UNMAPPED)
    assert body["model_type"] == "MODEL_TYPE_VOICE"
    assert body["phone_number"] == "sip:example-dental@sip.vapi.ai"
    assert body["metadata"] == {"audio_codec": "PCMU"}
    assert body["customer_agent_id"] == "vapi/pinned"
    assert body["attributes"]["platform"] == "vapi"
    assert body["tags"] == ["orchestration", "dental", "vapi"]
    assert body["prompt"].startswith("BrightSmile Dental")


def test_dial_target_is_required_and_named_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VAPI_ID_ROULETTE_DIAL_TARGET", raising=False)
    with pytest.raises(RuntimeError, match="VAPI_ID_ROULETTE_DIAL_TARGET is unset"):
        coval_agent_body(UNMAPPED)


def _coval_client(state: dict[str, Any]) -> CovalClient:
    def handler(request: httpx.Request) -> httpx.Response:
        state.setdefault("calls", []).append((request.method, request.url.path))
        if request.url.path == "/v1/agents" and request.method == "GET":
            wanted = request.url.params["filter"].split('"')[1]
            agents = [a for a in state["agents"] if a.get("customer_agent_id") == wanted]
            return httpx.Response(200, json={"agents": agents, "next_page_token": ""})
        if request.url.path == "/v1/agents":
            created = {**json.loads(request.content), "id": "A" * 22}
            state["agents"].append(created)
            return httpx.Response(200, json={"agent": created})
        if request.url.path.startswith("/v1/agents/") and request.method == "GET":
            wanted = request.url.path.rsplit("/", 1)[-1]
            agent = next(a for a in state["agents"] if a.get("id") == wanted)
            return httpx.Response(200, json={"agent": agent})
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
        agent_id, result = register(client, UNMAPPED)
    assert agent_id == "A" * 22
    assert set(result.update) == set(platform_assets.COVAL_MANAGED)
    assert state["agents"][0]["phone_number"] == "sip:example-dental@sip.vapi.ai"


def test_register_patches_only_the_drifted_fields(env: None) -> None:
    live = {**coval_agent_body(UNMAPPED), "id": "A", "phone_number": "sip:old@sip.vapi.ai"}
    state: dict[str, Any] = {"agents": [live]}
    with _coval_client(state) as client:
        agent_id, result = register(client, UNMAPPED)
    assert agent_id == "A"
    assert list(result.update) == ["phone_number"]
    assert state["patched"] == {"phone_number": "sip:example-dental@sip.vapi.ai"}


def test_register_matches_on_customer_agent_id_not_display_name(env: None) -> None:
    other = {**coval_agent_body(UNMAPPED), "id": "B", "customer_agent_id": "someone-else"}
    state: dict[str, Any] = {"agents": [other]}
    with _coval_client(state) as client:
        agent_id, _ = register(client, UNMAPPED)
    assert agent_id == "A" * 22
    assert len(state["agents"]) == 2


def test_register_dry_run_writes_nothing(env: None) -> None:
    state: dict[str, Any] = {"agents": []}
    with _coval_client(state) as client:
        agent_id, result = register(client, UNMAPPED, dry_run=True)
    assert agent_id == ""
    assert len(result.update) == len(platform_assets.COVAL_MANAGED) == 6
    assert all(method == "GET" for method, _ in state["calls"])


def test_launch_body_is_reproducible_and_tagged_with_the_arm(env: None) -> None:
    body = launch_body("A" * 22, VAPI, "P" * 22, "T" * 8, ("M" * 22,), 3, 42)
    assert body["agent_id"] == "A" * 22
    assert body["options"] == {
        "iteration_count": 1,
        "concurrency": 1,
        "sub_sample_size": 3,
        "sub_sample_seed": 42,
    }
    assert body["metric_ids"] == ["M" * 22]
    assert body["metadata"]["customer_metadata"]["arm"] == "vapi/pinned"
    assert body["metadata"]["display_name"].startswith("vapi/pinned e2e ")


def test_launch_run_unwraps_the_run(env: None) -> None:
    state: dict[str, Any] = {"agents": []}
    with _coval_client(state) as client:
        run = client.launch_run(launch_body("A" * 22, VAPI, "P" * 22, "T" * 8, (), 1, 1))
    assert run == {"run_id": "R" * 22, "status": "PENDING"}
    assert state["launched"]["options"]["sub_sample_size"] == 1


def test_drift_lists_what_apply_would_still_change_without_writing(env: None) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(200, json=LIVE)

    with _client(handler) as client:
        assert drift(client, VAPI, BASE) == ["model.tools"]
    assert seen == ["GET"]


def test_drift_is_empty_once_the_platform_matches(env: None) -> None:
    synced = {**LIVE, "model": {**LIVE["model"], "tools": desired(VAPI, BASE)["model.tools"]}}
    with _client(lambda r: httpx.Response(200, json=synced)) as client:
        assert drift(client, VAPI, BASE) == []


def test_register_reconciles_display_name_and_tags_too(env: None) -> None:
    live = {**coval_agent_body(UNMAPPED), "id": "A", "display_name": "old name", "tags": []}
    state: dict[str, Any] = {"agents": [live]}
    with _coval_client(state) as client:
        _, result = register(client, UNMAPPED)
    assert sorted(result.update) == ["display_name", "tags"]
    assert state["patched"] == {
        "display_name": "vapi/pinned",
        "tags": ["orchestration", "dental", "vapi"],
    }


def test_register_refuses_a_non_voice_agent_with_our_customer_id(env: None) -> None:
    live = {**coval_agent_body(UNMAPPED), "id": "A", "model_type": "MODEL_TYPE_SMS"}
    with (
        _coval_client({"agents": [live]}) as client,
        pytest.raises(SyncError, match="MODEL_TYPE_SMS"),
    ):
        register(client, UNMAPPED)


def test_contract_hash_refuses_to_label_without_the_fixtures(
    env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform_assets, "has_private_contract", lambda suite: False)
    with pytest.raises(SyncError, match="seeded world"):
        coval_agent_body(UNMAPPED)
    with pytest.raises(SyncError, match="seeded world"):
        launch_body("A" * 22, VAPI, "P" * 22, "T" * 8, (), 1, 1)


# --- every platform, one case each ------------------------------------------

from dataclasses import dataclass  # noqa: E402

from coval_bench.mocktools.codecs import (  # noqa: E402
    PRESET_CALLER,
    PRESET_SIMULATION,
    Correlation,
    codec_for,
)
from coval_bench.platform_assets import (  # noqa: E402
    RETELL_LLM_FIELDS,
    TELNYX_SECRETS,
    RetellClient,
    TelnyxClient,
    prepare_retell,
    prepare_telnyx,
    retell_number,
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

RETELL_AGENT: dict[str, Any] = {
    "agent_id": "agent_1",
    "agent_name": "Retell Dental",
    "version": 4,
    "is_published": False,
    "response_engine": {"type": "retell-llm", "llm_id": "llm_1", "version": None},
    "voice_id": "11labs-Cimo",
    "voice_model": "eleven_flash_v2",
    "language": "en-US",
    "stt_mode": "fast",
    "post_call_analysis_model": None,
}

RETELL_LLM: dict[str, Any] = {
    "llm_id": "llm_1",
    "version": 2,
    "model": "gpt-4.1",
    "model_temperature": 0,
    "general_prompt": PROMPT,
    "begin_message": None,
    "general_tools": [{"type": "end_call", "name": "end_call", "description": "End the call."}],
}

RETELL_NUMBER = "+14045550142"
RETELL_DIAL = f"sip:{RETELL_NUMBER}@sip.retellai.com"
RETELL_ROUTE = [{"agent_id": "agent_1", "agent_version": "latest", "weight": 1}]


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
    update_path: str | None = None
    view: dict[str, Any] | None = None


CASES = [
    Case(
        key="vapi/pinned",
        env={"VAPI_ID_ROULETTE_ASSISTANT_ID": "asst_1", "VAPI_API_KEY": "vapi-key"},
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
        key="telnyx/pinned",
        env={
            "TELNYX_ID_ROULETTE_ASSISTANT_ID": "assistant-1",
            "TELNYX_API_KEY": "telnyx-key",
            "TELNYX_ID_ROULETTE_DIAL_TARGET": "sip:dental@example-dental.sip.telnyx.com",
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
                "instructions",
                "greeting",
                "llm_api_key_ref",
                "transcription.model",
                "voice_settings.voice",
                "voice_settings.api_key_ref",
                "voice_settings.expressive_mode",
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
    Case(
        key="retell/pinned",
        env={
            "RETELL_ID_ROULETTE_AGENT_ID": "agent_1",
            "RETELL_API_KEY": "retell-key",
            "RETELL_ID_ROULETTE_DIAL_TARGET": RETELL_DIAL,
        },
        live=RETELL_AGENT,
        agent_path="/get-agent/agent_1",
        update_method="PATCH",
        update_path="/update-retell-llm/llm_1",
        tool_name=lambda t: t["name"],
        tool_url=lambda t: t["url"],
        secret_in_config=True,
        expected_drift=frozenset({"general_tools"}),
        pins={
            "model": "gpt-4.1",
            "model_temperature": 0,
            "voice_model": "eleven_flash_v2",
            "voice_id": "11labs-Cimo",
            "stt_mode": "fast",
            "language": "en-US",
            "post_call_analysis_model": None,
        },
        wraps_in_data=False,
        view={**RETELL_AGENT, **{k: RETELL_LLM[k] for k in RETELL_LLM_FIELDS}},
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


CLIENTS: dict[str, Any] = {"vapi": VapiClient, "telnyx": TelnyxClient, "retell": RetellClient}


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
        if request.url.path == "/get-retell-llm/llm_1" and request.method == "GET":
            return httpx.Response(200, json=state.get("llm", RETELL_LLM))
        if request.url.path == "/update-retell-llm/llm_1":
            state["patched"] = json.loads(request.content)
            state["llm"] = {**state.get("llm", RETELL_LLM), **state["patched"]}
            return httpx.Response(200, json=state["llm"])
        if request.url.path == "/update-agent/agent_1":
            state["agent_patched"] = json.loads(request.content)
            state["live"] = {**state.get("live", case.live), **state["agent_patched"]}
            return httpx.Response(200, json=state["live"])
        if request.url.path == f"/get-phone-number/{RETELL_NUMBER}":
            return httpx.Response(200, json={"inbound_agents": state["route"]})
        if request.url.path == f"/update-phone-number/{RETELL_NUMBER}":
            state["route"] = json.loads(request.content)["inbound_agents"]
            return httpx.Response(200, json={"inbound_agents": state["route"]})
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
    state.setdefault("sub", "example-dental")
    state.setdefault("recv", "from_anyone")
    state.setdefault("route", RETELL_ROUTE)
    return CLIENTS[spec.platform]("key", platform.api_base, transport=httpx.MockTransport(handler))


def test_every_platform_renders_one_tool_per_contract_entry_in_order(case: Case) -> None:
    spec = spec_for(case.key)
    tools = desired(spec, BASE)[platform_for(spec).tools_path]
    assert [case.tool_name(t) for t in tools] == [d["name"] for d in CONTRACT]
    assert len(tools) == len(CONTRACT)


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
    result = plan(case.view or case.live, desired(spec, BASE), platform_for(spec).canon)
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
        assert (case.update_method, case.update_path or case.agent_path) in state["calls"]
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
    chosen = next(c for c in CASES if c.key == "telnyx/pinned")
    for name, value in chosen.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MOCK_TOOLS_SECRET", SECRET)
    return chosen


def test_telnyx_correlation_rides_in_preset_fields_the_model_cannot_see(telnyx_env: Case) -> None:
    (tool, *_) = desired(spec_for("telnyx/pinned"), BASE)["tools"]
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
        "sip:dental@example-dental.sip.telnyx.com",
        "SIP:x@Example-Dental.sip.telnyx.com:5060",
    ],
)
def test_sip_subdomain_is_read_off_the_dial_target(target: str) -> None:
    assert sip_subdomain(target) == "example-dental"


@pytest.mark.parametrize(
    "target", ["sip:dental@sip.vapi.ai", "+14045550710", "coval.sip.telnyx.com"]
)
def test_sip_subdomain_refuses_anything_else(target: str) -> None:
    with pytest.raises(SyncError):
        sip_subdomain(target)


def test_telnyx_prepare_lists_missing_secrets_and_subdomain_then_creates_only_those(
    telnyx_env: Case,
) -> None:
    spec = spec_for("telnyx/pinned")
    state: dict[str, Any] = {"secrets": ["coval", "coval-bench-openai"], "sub": None}
    with _platform_client(telnyx_env, state) as client:
        pending = prepare_telnyx(client, spec, True)
        assert pending == [
            "integration_secret:coval-bench-elevenlabs",
            "integration_secret:coval-bench-mock",
            "sip_subdomain:texml-1=example-dental:from_anyone",
        ]
        assert state["secrets"] == ["coval", "coval-bench-openai"] and state["sub"] is None
        assert prepare_telnyx(client, spec, False) == pending
        assert prepare_telnyx(client, spec, False) == []
    assert set(state["secrets"]) == {"coval", *TELNYX_SECRETS}
    assert state["sub"] == "example-dental"


def test_telnyx_prepare_refuses_a_foreign_client(telnyx_env: Case) -> None:
    with pytest.raises(SyncError), _client(lambda r: httpx.Response(200, json={})) as vapi:
        prepare_telnyx(vapi, spec_for("telnyx/pinned"), True)


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
    spec = spec_for("telnyx/pinned")
    state: dict[str, Any] = {"sub": "example-dental", "recv": "only_my_connections"}
    with _platform_client(telnyx_env, state) as client:
        assert prepare_telnyx(client, spec, True) == [
            "sip_subdomain:texml-1=example-dental:from_anyone"
        ]
        prepare_telnyx(client, spec, False)
        assert prepare_telnyx(client, spec, False) == []
    assert state["recv"] == "from_anyone"


def test_telnyx_canon_ignores_what_the_server_adds_to_tools(telnyx_env: Case) -> None:
    from coval_bench.platform_assets import TELNYX_CANON

    (tool, *_) = desired(spec_for("telnyx/pinned"), BASE)["tools"]
    echoed = [{**tool, "tool_id": "tool-1", "shared": False, "timeout_ms": 5000}]
    assert TELNYX_CANON["tools"](echoed) == [tool]
    assert TELNYX_CANON["tool_ids"](None) == []
    assert TELNYX_CANON["tool_ids"](["t"]) == ["t"]


# --- retell only: the agent and its LLM are two objects --------------------


@pytest.fixture
def retell_env(monkeypatch: pytest.MonkeyPatch) -> Case:
    chosen = next(c for c in CASES if c.key == "retell/pinned")
    for name, value in chosen.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MOCK_TOOLS_SECRET", SECRET)
    return chosen


def test_retell_get_agent_merges_the_llm_fields_the_agent_only_points_at(
    retell_env: Case,
) -> None:
    state: dict[str, Any] = {}
    with _platform_client(retell_env, state) as client:
        live = client.get_agent("agent_1")
    assert state["calls"] == [("GET", "/get-agent/agent_1"), ("GET", "/get-retell-llm/llm_1")]
    assert live["voice_id"] == "11labs-Cimo"
    assert live["general_prompt"] == PROMPT
    assert live["general_tools"] == RETELL_LLM["general_tools"]
    assert live["version"] == RETELL_AGENT["version"]


def test_retell_apply_patches_only_the_llm_when_only_tools_drift(retell_env: Case) -> None:
    spec = spec_for("retell/pinned")
    state: dict[str, Any] = {}
    with _platform_client(retell_env, state) as client:
        apply(client, spec, desired(spec, BASE))
    assert [path for _, path in state["calls"]] == [
        "/get-agent/agent_1",
        "/get-retell-llm/llm_1",
        f"/get-phone-number/{RETELL_NUMBER}",
        "/update-retell-llm/llm_1",
    ]
    assert set(state["patched"]) == {"general_tools"}
    assert len(state["patched"]["general_tools"]) == len(CONTRACT)
    assert "agent_patched" not in state


def test_retell_update_routes_each_field_to_its_owner(retell_env: Case) -> None:
    state: dict[str, Any] = {}
    with _platform_client(retell_env, state) as client:
        merged = client.update_agent("agent_1", {"general_tools": [], "voice_id": "11labs-Kate"})
    assert state["patched"] == {"general_tools": []}
    assert state["agent_patched"] == {"voice_id": "11labs-Kate"}
    assert merged["general_tools"] == [] and merged["voice_id"] == "11labs-Kate"
    assert {"general_tools", "general_prompt", "begin_message"} <= RETELL_LLM_FIELDS


def test_retell_refuses_an_agent_that_does_not_answer_with_a_retell_llm(retell_env: Case) -> None:
    engine = {"type": "conversation-flow", "conversation_flow_id": "cf_1"}
    state: dict[str, Any] = {"live": {**RETELL_AGENT, "response_engine": engine}}
    with (
        _platform_client(retell_env, state) as client,
        pytest.raises(SyncError, match="'conversation-flow'; only retell-llm"),
    ):
        client.get_agent("agent_1")


def test_retell_follows_the_llm_version_the_agent_pins(retell_env: Case) -> None:
    engine = {**RETELL_AGENT["response_engine"], "version": 3}
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "retell-llm" in request.url.path:
            seen.append(dict(request.url.params))
            return httpx.Response(200, json=RETELL_LLM)
        return httpx.Response(200, json={**RETELL_AGENT, "response_engine": engine})

    transport = httpx.MockTransport(handler)
    with RetellClient("k", "https://api.retellai.com", transport=transport) as client:
        client.get_agent("agent_1")
        client.update_agent("agent_1", {"general_tools": []})
    assert seen == [{"version": "3"}, {"version": "3"}]


def test_retell_tools_speak_no_filler_and_never_retry(retell_env: Case) -> None:
    (tool, *_) = desired(spec_for("retell/pinned"), BASE)["general_tools"]
    assert tool["type"] == "custom" and tool["method"] == "POST"
    assert tool["args_at_root"] is True
    assert tool["speak_during_execution"] is False
    assert tool["max_retry"] == 0
    assert tool["headers"] == {
        "X-Mock-Tools-Key": SECRET,
        "X-Coval-Simulation-Id": "{{coval-simulation-id}}",
        "X-Coval-Caller-Number": "{{user_number}}",
    }


def test_retell_apply_sends_each_owner_only_what_drifted(retell_env: Case) -> None:
    spec = spec_for("retell/pinned")
    state: dict[str, Any] = {"live": {**RETELL_AGENT, "voice_model": "eleven_turbo_v2"}}
    with _platform_client(retell_env, state) as client:
        result = apply(client, spec, desired(spec, BASE))
    assert set(result.update) == {"general_tools", "voice_model"}
    assert set(state["patched"]) == {"general_tools"}
    assert state["agent_patched"] == {"voice_model": "eleven_flash_v2"}


def test_retell_refuses_a_multi_state_llm(retell_env: Case) -> None:
    llm = {**RETELL_LLM, "states": [{"name": "book", "state_prompt": "Book.", "tools": []}]}
    state: dict[str, Any] = {"llm": llm}
    with (
        _platform_client(retell_env, state) as client,
        pytest.raises(SyncError, match="has 1 states"),
    ):
        client.get_agent("agent_1")


def test_retell_prepare_reports_then_binds_the_number_to_the_newest_draft(
    retell_env: Case,
) -> None:
    spec = spec_for("retell/pinned")
    state: dict[str, Any] = {"route": []}
    with _platform_client(retell_env, state) as client:
        pending = prepare_retell(client, spec, True)
        assert pending == [f"route:{RETELL_NUMBER}=agent_1@latest"]
        assert state["route"] == []
        assert prepare_retell(client, spec, False) == pending
        assert prepare_retell(client, spec, False) == []
    assert state["route"] == RETELL_ROUTE


def test_retell_prepare_flags_a_number_pinned_to_a_published_version(retell_env: Case) -> None:
    pinned = [{"agent_id": "agent_1", "agent_version": 3, "weight": 1}]
    with _platform_client(retell_env, {"route": pinned}) as client:
        assert prepare_retell(client, spec_for("retell/pinned"), True) == [
            f"route:{RETELL_NUMBER}=agent_1@latest"
        ]


def test_retell_prepare_never_repoints_someone_elses_number(retell_env: Case) -> None:
    theirs = [{"agent_id": "agent_9", "agent_version": "latest", "weight": 1}]
    with (
        _platform_client(retell_env, {"route": theirs}) as client,
        pytest.raises(SyncError, match="agent_9.*not ours to repoint"),
    ):
        prepare_retell(client, spec_for("retell/pinned"), True)


@pytest.mark.parametrize(
    "target", ["sip:+14045550142@sip.retellai.com", "SIP:+14045550142@Sip.RetellAI.com:5060"]
)
def test_retell_number_is_read_off_the_sip_dial_target(target: str) -> None:
    assert retell_number(target) == "+14045550142"


@pytest.mark.parametrize(
    "target",
    [
        "+14045550142",
        "sip:dental@sip.retellai.com",
        "sip:+14045550142@sip.vapi.ai",
        "sip:14045550142@sip.retellai.com",
    ],
)
def test_retell_number_refuses_anything_else(target: str) -> None:
    with pytest.raises(SyncError):
        retell_number(target)


def test_launch_body_selects_exact_cases_instead_of_a_sample(env: None) -> None:
    body = launch_body("A" * 22, VAPI, "P" * 22, "T" * 8, (), 1, 42, ("C" * 22,))
    assert body["options"] == {
        "iteration_count": 1,
        "concurrency": 1,
        "test_case_ids": ["C" * 22],
    }


def test_retell_apply_writes_nothing_when_the_agent_is_rejected(retell_env: Case) -> None:
    spec = spec_for("retell/pinned")
    llm = {**RETELL_LLM, "states": [{"name": "book", "state_prompt": "Book.", "tools": []}]}
    state: dict[str, Any] = {"llm": llm, "route": []}
    with (
        _platform_client(retell_env, state) as client,
        pytest.raises(SyncError, match="has 1 states"),
    ):
        apply(client, spec, desired(spec, BASE))
    assert state["route"] == []
    assert "patched" not in state and "agent_patched" not in state
    assert all(method == "GET" for method, _ in state["calls"])


# --- asset-owned Coval agents ----------------------------------------------


def test_register_leaves_an_asset_owned_agent_to_terraform(env: None) -> None:
    with (
        _coval_client({"agents": []}) as client,
        pytest.raises(SyncError, match="only its dial target"),
    ):
        register(client, VAPI)


def test_only_the_dial_target_is_set_on_an_asset_owned_agent(env: None) -> None:
    live = {"id": "AGENT2", "phone_number": "+15550100000", "display_name": "from terraform"}
    state: dict[str, Any] = {"agents": [live]}
    with _coval_client(state) as client:
        assert sorted(register_dial_target(client, VAPI, "AGENT2", dry_run=True).update) == [
            "phone_number"
        ]
        assert "patched" not in state
        register_dial_target(client, VAPI, "AGENT2")
    assert state["patched"] == {"phone_number": "sip:example-dental@sip.vapi.ai"}


# --- the prompt ------------------------------------------------------------


def test_vapi_gets_the_suite_prompt_and_generates_its_greeting_from_it(env: None) -> None:
    dental = {
        **LIVE,
        "model": {**LIVE["model"], "messages": [{"role": "system", "content": "Front desk."}]},
        "firstMessageMode": "assistant-speaks-first",
        "firstMessage": "Thanks for calling BrightSmile.",
    }
    result = plan(dental, desired(VAPI, BASE))
    assert sorted(result.update) == ["firstMessageMode", "model.messages", "model.tools"]
    body = patch_body(dental, desired(VAPI, BASE))
    assert body["model"]["messages"] == [{"role": "system", "content": PROMPT}]
    assert body["model"]["provider"] == "openai"


def test_telnyx_gets_the_suite_prompt_and_generates_its_greeting_from_it(
    telnyx_env: Case,
) -> None:
    wanted = desired(spec_for("telnyx/pinned"), BASE)
    assert wanted["instructions"] == PROMPT
    assert wanted["greeting"] == "<assistant-speaks-first-with-model-generated-message>"


def test_retell_gets_the_suite_prompt_and_no_fixed_opening_line(retell_env: Case) -> None:
    wanted = desired(spec_for("retell/pinned"), BASE)
    assert wanted["general_prompt"] == PROMPT
    assert wanted["begin_message"] is None


def test_a_plan_never_prints_the_dial_target(capsys: pytest.CaptureFixture[str]) -> None:
    platform_assets._echo_plan(
        Plan(update={"phone_number": ("+15550100000", "sip:secret-target@sip.vapi.ai")})
    )
    printed = capsys.readouterr().out
    assert "secret-target" not in printed
    assert "+15550100000" not in printed
    assert "[REDACTED]" in printed


def test_prepare_steps_are_printed_by_kind_without_the_dial_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    platform_assets._echo_plan(Plan(prepared=["route:+15550100042=agent_1@latest"]))
    printed = capsys.readouterr().out
    assert "+15550100042" not in printed
    assert "route:[REDACTED]" in printed
