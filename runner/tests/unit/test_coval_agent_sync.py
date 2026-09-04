# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The sync-llm command and the Phonely text agent it defines."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from click.testing import CliRunner
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.llm import coval_agent
from coval_bench.llm.coval_agent import (
    CUSTOMER_AGENT_ID,
    RUN_NAME,
    CovalTextAgentDefinition,
    CovalTextClient,
    sync,
    sync_llm,
)
from coval_bench.platform_assets import SyncError

SECRET = "proxy-secret-value"  # noqa: S105
DEFINITION = CovalTextAgentDefinition(
    proxy_url="https://api.example.com",
    proxy_secret=SecretStr(SECRET),
    test_set_id="TSDENTAL",
    instruction_metric_id="M" * 22,
)


def _state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "agents": [],
        "test_set_agents": [],
        "run_templates": [],
        "scheduled_runs": [],
        "writes": [],
        "filters": [],
    }
    state.update(overrides)
    return state


def _client(state: dict[str, Any]) -> CovalTextClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        if request.method == "GET" and path == "/agents":
            state["filters"].append(dict(request.url.params))
            wanted = request.url.params["filter"].split('"')[1]
            agents = [a for a in state["agents"] if a.get("customer_agent_id") == wanted]
            return httpx.Response(200, json={"agents": agents})
        if request.method == "GET":
            state_key, response_key = {
                "/agents": ("agents", "agents"),
                "/test-sets/TSDENTAL/agents": ("test_set_agents", "agents"),
                "/run-templates": ("run_templates", "run_templates"),
                "/scheduled-runs": ("scheduled_runs", "scheduled_runs"),
            }[path]
            return httpx.Response(200, json={response_key: state[state_key]})
        state["writes"].append((path, body))
        if path == "/agents":
            created = {**body, "id": "A" * 22}
            state["agents"].append(created)
            return httpx.Response(200, json={"agent": created})
        if path.startswith("/agents/"):
            return httpx.Response(200, json={"agent": {**state["agents"][0], **body}})
        if path == "/test-sets/TSDENTAL/agents:add":
            state["test_set_agents"].extend({"id": i} for i in body["agent_ids"])
            return httpx.Response(200, json={})
        if path == "/run-templates":
            created = {**body, "id": "T" * 22}
            state["run_templates"].append(created)
            return httpx.Response(200, json={"run_template": created})
        if path == "/scheduled-runs":
            return httpx.Response(200, json={"scheduled_run": {**body, "id": "S" * 22}})
        return httpx.Response(
            400, json={"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": path}}
        )

    return CovalTextClient("coval-key", "https://api.coval.dev/v1", httpx.MockTransport(handler))


def test_body_renders_the_proxy_contract_and_survives_covals_substitution() -> None:
    body = DEFINITION.agent_body()
    assert body["customer_agent_id"] == CUSTOMER_AGENT_ID
    assert body["model_type"] == "MODEL_TYPE_CHAT"
    assert body["metadata"]["chat_endpoint"] == "https://api.example.com/llm/phonely/chat"
    assert body["metadata"]["authorization_header"] == f"Bearer {SECRET}"
    substituted = (
        body["metadata"]["input_template"]
        .replace("{{session_id}}", "call-1")
        .replace("{{messages}}", '[{"role": "user", "content": "hi"}]')
        .replace("{{simulation_output_id}}", "sim-1")
    )
    assert json.loads(substituted) == {
        "model": "call-1",
        "messages": [{"role": "user", "content": "hi"}],
        "simulation_id": "sim-1",
    }
    assert SECRET not in json.dumps(DEFINITION.redacted_body())


def test_from_settings_names_every_missing_setting() -> None:
    with pytest.raises(SyncError, match="llm_proxy_public_url, llm_proxy_secret"):
        CovalTextAgentDefinition.from_settings(
            Settings(coval_s2s_dental_test_set_id="T", coval_s2s_instruction_metric_id="M")
        )
    definition = CovalTextAgentDefinition.from_settings(
        Settings(
            llm_proxy_public_url="https://api.example.com/",
            llm_proxy_secret=SecretStr(SECRET),
            coval_s2s_instruction_metric_id="M",
        ),
        test_set_id="T",
    )
    assert (definition.proxy_url, definition.test_set_id) == ("https://api.example.com", "T")


def test_sync_creates_everything_when_absent() -> None:
    state = _state()
    with _client(state) as client:
        result = sync(client, DEFINITION)

    assert result.agent_id == "A" * 22
    assert result.actions == [
        "agent: create",
        "test set: attach",
        "run template: create",
        "scheduled run: create",
    ]
    assert [path for path, _ in state["writes"]] == [
        "/agents",
        "/test-sets/TSDENTAL/agents:add",
        "/run-templates",
        "/scheduled-runs",
    ]
    template = state["writes"][2][1]
    assert template["agent_ids"] == ["A" * 22]
    assert template["persona_ids"] == list(coval_agent.CLEAN_DENTAL_PERSONAS)
    assert template["test_set_ids"] == ["TSDENTAL"]
    assert template["metric_ids"] == ["M" * 22]
    assert template["iteration_count"] == 1
    assert "options" not in template
    assert state["writes"][3][1]["schedule_expression"] == "cron(0 13 * * ? *)"


def test_sync_patches_drifted_metadata_wholesale_and_leaves_the_rest() -> None:
    live = {**DEFINITION.agent_body(), "id": "A"}
    live["metadata"] = {**live["metadata"], "authorization_header": "Bearer stale"}
    state = _state(
        agents=[live],
        test_set_agents=[{"id": "A"}],
        run_templates=[{"id": "T", "display_name": RUN_NAME}],
        scheduled_runs=[{"id": "S", "run_template_id": "T"}],
    )
    with _client(state) as client:
        result = sync(client, DEFINITION)

    assert result.actions == [
        "agent: patch ['metadata']",
        "test set: attached",
        "run template: exists",
        "scheduled run: exists",
    ]
    assert state["writes"] == [("/agents/A", {"metadata": DEFINITION.agent_body()["metadata"]})]


def test_sync_looks_up_by_customer_id_filter_and_never_adopts_a_name_only_match() -> None:
    name_only = {**DEFINITION.agent_body(), "id": "B", "customer_agent_id": ""}
    state = _state(agents=[name_only])
    with _client(state) as client:
        result = sync(client, DEFINITION)
    assert state["filters"] == [
        {"filter": f'customer_agent_id="{CUSTOMER_AGENT_ID}"', "page_size": "1"}
    ]
    assert result.agent_id == "A" * 22
    assert state["writes"][0][0] == "/agents"


def test_sync_dry_run_reports_and_writes_nothing() -> None:
    state = _state()
    with _client(state) as client:
        result = sync(client, DEFINITION, dry_run=True)
    assert result.agent_id == ""
    assert result.actions[0] == "agent: create"
    assert state["writes"] == []


def test_sync_refuses_a_foreign_model_type_and_surfaces_api_errors() -> None:
    live = {**DEFINITION.agent_body(), "id": "A", "model_type": "MODEL_TYPE_VOICE"}
    with (
        _client(_state(agents=[live])) as client,
        pytest.raises(SyncError, match="MODEL_TYPE_VOICE"),
    ):
        sync(client, DEFINITION)

    with (
        _client(_state(test_set_agents=None)) as client,
        pytest.raises(SyncError, match="INVALID_ARGUMENT") as failure,
    ):
        client.add_test_set_agents("OTHER", ["A"])
    assert "agents:add" in str(failure.value)


def test_cli_prints_the_agent_id_and_never_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state()
    monkeypatch.setenv("COVAL_API_KEY", "coval-key")
    monkeypatch.setattr(
        coval_agent,
        "get_settings",
        lambda: Settings(
            llm_proxy_public_url="https://api.example.com",
            llm_proxy_secret=SecretStr(SECRET),
            coval_s2s_dental_test_set_id="TSDENTAL",
            coval_s2s_instruction_metric_id="M" * 22,
        ),
    )
    monkeypatch.setattr(coval_agent, "CovalTextClient", lambda *_args: _client(state))

    dry = CliRunner().invoke(sync_llm, ["--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "COVAL_LLM_PHONELY_AGENT_ID=<created on apply>" in dry.output
    assert state["writes"] == []

    applied = CliRunner().invoke(sync_llm, [])
    assert applied.exit_code == 0, applied.output
    assert f"COVAL_LLM_PHONELY_AGENT_ID={'A' * 22}" in applied.output
    assert "llm_sync_fetch_deferred" in applied.output
    assert SECRET not in dry.output + applied.output


def test_cli_syncs_completed_runs_into_the_database(monkeypatch: pytest.MonkeyPatch) -> None:
    agent_id = "A" * 22
    template_id = "T" * 22
    state = _state(
        agents=[{**DEFINITION.agent_body(), "id": agent_id}],
        test_set_agents=[{"id": agent_id}],
        run_templates=[{"id": template_id, "display_name": RUN_NAME}],
        scheduled_runs=[{"id": "S" * 22, "run_template_id": template_id}],
    )
    settings = Settings(
        llm_proxy_public_url="https://api.example.com",
        llm_proxy_secret=SecretStr(SECRET),
        coval_s2s_dental_test_set_id="TSDENTAL",
        coval_s2s_instruction_metric_id="M" * 22,
    )
    fetch = MagicMock()
    monkeypatch.setenv("COVAL_API_KEY", "coval-key")
    monkeypatch.setattr(coval_agent, "get_settings", lambda: settings)
    monkeypatch.setattr(coval_agent, "CovalTextClient", lambda *_args: _client(state))
    monkeypatch.setattr("coval_bench.s2s.fetch_v2v._run_fetch", fetch)

    applied = CliRunner().invoke(sync_llm, [])

    assert applied.exit_code == 0, applied.output
    fetch.assert_called_once()
    assert fetch.call_args.kwargs["settings"].coval_llm_phonely_agent_id == agent_id


def test_cli_logs_automation_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coval_agent, "get_settings", Settings)

    failed = CliRunner().invoke(sync_llm, [])

    assert failed.exit_code == 1
    assert '"event": "RUN_FAILED"' in failed.output
    assert "sync-llm needs" in failed.output
