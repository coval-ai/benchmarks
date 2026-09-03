# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The Coval side of the Phonely text agent, defined in code and pushed idempotently."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Self

import click
import httpx
from pydantic import BaseModel, SecretStr

from coval_bench.config import Settings, get_settings
from coval_bench.platform_assets import COVAL_API_BASE, COVAL_API_KEY, CovalClient, SyncError, plan
from coval_bench.variants.platforms import redact

CUSTOMER_AGENT_ID = "benchmarks-phonely-text"
DISPLAY_NAME = "Benchmarks: Phonely text agent"
# The public API rejects MODEL_TYPE_TEXT; CHAT is the HTTP text simulator.
MODEL_TYPE = "MODEL_TYPE_CHAT"
RUN_NAME = "benchmarks-phonely-text-daily"
CLEAN_DENTAL_PERSONAS = ("PN3xgmsqeLDjsNNEA2e55e", "9ATy64zKXxSUaVWb5YnQtd")
SCHEDULE_EXPRESSION = "cron(0 13 * * ? *)"
SCHEDULE_TIMEZONE = "UTC"
INPUT_TEMPLATE = (
    '{"model": "{{session_id}}", "messages": {{messages}}, '
    '"simulation_id": "{{simulation_output_id}}"}'
)
# PATCH replaces metadata wholesale, so it is reconciled as one value.
MANAGED = ("display_name", "metadata", "test_set_ids")


class CovalTextAgentDefinition(BaseModel, frozen=True):
    proxy_url: str
    proxy_secret: SecretStr
    test_set_id: str
    instruction_metric_id: str

    @classmethod
    def from_settings(cls, settings: Settings, *, test_set_id: str | None = None) -> Self:
        proxy_url = settings.llm_proxy_public_url
        proxy_secret = settings.llm_proxy_secret
        dental = test_set_id or settings.coval_s2s_dental_test_set_id
        metric = settings.coval_s2s_instruction_metric_id
        missing = [
            name
            for name, value in (
                ("llm_proxy_public_url", proxy_url),
                ("llm_proxy_secret", proxy_secret),
                ("coval_s2s_dental_test_set_id", dental),
                ("coval_s2s_instruction_metric_id", metric),
            )
            if not value
        ]
        if missing or proxy_url is None or proxy_secret is None or not dental or not metric:
            raise SyncError(f"sync-coval needs {', '.join(missing)} set")
        return cls(
            proxy_url=proxy_url.rstrip("/"),
            proxy_secret=proxy_secret,
            test_set_id=dental,
            instruction_metric_id=metric,
        )

    def agent_body(self) -> dict[str, Any]:
        return {
            "display_name": DISPLAY_NAME,
            "customer_agent_id": CUSTOMER_AGENT_ID,
            "model_type": MODEL_TYPE,
            "metadata": {
                "chat_endpoint": f"{self.proxy_url}/llm/phonely/chat",
                "initialization_endpoint": f"{self.proxy_url}/llm/phonely/session",
                "initialization_payload": "{}",
                "authorization_header": f"Bearer {self.proxy_secret.get_secret_value()}",
                "input_template": INPUT_TEMPLATE,
                "response_message_path": "choices[0].message.content",
                "response_format": "chat_completions",
                "response_stream_format": "none",
                "strip_message_timestamps": True,
                "tool_call_extraction": {
                    "enabled": True,
                    "tool_calls_path": "choices[].message.tool_calls[]",
                    "tool_call_mappings": {
                        "id": "id",
                        "name": "function.name",
                        "arguments": "function.arguments",
                    },
                },
            },
            "test_set_ids": [self.test_set_id],
        }

    def redacted_body(self) -> dict[str, Any]:
        found: list[str] = []
        redacted: dict[str, Any] = redact(self.agent_body(), found)
        return redacted

    def run_template_body(self, agent_id: str) -> dict[str, Any]:
        return {
            "display_name": RUN_NAME,
            "agent_id": agent_id,
            "persona_ids": list(CLEAN_DENTAL_PERSONAS),
            "test_set_id": self.test_set_id,
            "metric_ids": [self.instruction_metric_id],
            "options": {"iteration_count": 1},
        }


def scheduled_run_body(run_template_id: str) -> dict[str, Any]:
    return {
        "display_name": RUN_NAME,
        "run_template_id": run_template_id,
        "schedule_expression": SCHEDULE_EXPRESSION,
        "schedule_timezone": SCHEDULE_TIMEZONE,
        "enabled": True,
    }


class CovalTextClient(CovalClient):
    def __enter__(self) -> Self:
        return self

    def find_agent(self, customer_agent_id: str) -> dict[str, Any] | None:
        by_name = None
        for agent in self._pages("/agents", "agents"):
            if agent.get("customer_agent_id") == customer_agent_id:
                return agent
            if by_name is None and agent.get("display_name") == DISPLAY_NAME:
                by_name = agent
        return by_name

    def test_set_agent_ids(self, test_set_id: str) -> set[str]:
        return {
            str(agent["id"])
            for agent in self._pages(f"/test-sets/{test_set_id}/agents", "agents")
            if agent.get("id")
        }

    def add_test_set_agents(self, test_set_id: str, agent_ids: list[str]) -> None:
        self._request("POST", f"/test-sets/{test_set_id}/agents:add", {"agent_ids": agent_ids})

    def find_run_template(self, display_name: str) -> dict[str, Any] | None:
        for template in self._pages("/run-templates", "run_templates"):
            if template.get("display_name") == display_name:
                return template
        return None

    def create_run_template(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("POST", "/run-templates", body)
        template = payload.get("run_template")
        return template if isinstance(template, dict) else payload

    def find_scheduled_run(self, run_template_id: str) -> dict[str, Any] | None:
        for scheduled in self._pages("/scheduled-runs", "scheduled_runs"):
            if scheduled.get("run_template_id") == run_template_id:
                return scheduled
        return None

    def create_scheduled_run(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("POST", "/scheduled-runs", body)
        scheduled = payload.get("scheduled_run")
        return scheduled if isinstance(scheduled, dict) else payload


@dataclass
class SyncResult:
    agent_id: str = ""
    actions: list[str] = field(default_factory=list)


def sync(
    client: CovalTextClient, definition: CovalTextAgentDefinition, *, dry_run: bool = False
) -> SyncResult:
    """Find-or-create the agent, its test-set link, run template, and schedule."""
    result = SyncResult()
    wanted = definition.agent_body()
    live = client.find_agent(CUSTOMER_AGENT_ID)
    if live is None:
        result.actions.append("agent: create")
        if dry_run:
            result.actions.append("test set: attach")
            result.actions.append("run template: create")
            result.actions.append("scheduled run: create")
            return result
        result.agent_id = str(client.create_agent(wanted)["id"])
    else:
        if live.get("model_type") != MODEL_TYPE:
            raise SyncError(
                f"coval agent {live.get('id')} is {live.get('model_type')!r}, not {MODEL_TYPE}; "
                "model_type cannot be patched, so this record is not ours to reconcile"
            )
        result.agent_id = str(live["id"])
        drift = plan(live, {path: wanted[path] for path in MANAGED})
        if drift.update:
            result.actions.append(f"agent: patch {sorted(drift.update)}")
            if not dry_run:
                client.update_agent(result.agent_id, {path: wanted[path] for path in drift.update})
        else:
            result.actions.append("agent: unchanged")

    if result.agent_id in client.test_set_agent_ids(definition.test_set_id):
        result.actions.append("test set: attached")
    else:
        result.actions.append("test set: attach")
        if not dry_run:
            client.add_test_set_agents(definition.test_set_id, [result.agent_id])

    template = client.find_run_template(RUN_NAME)
    if template is None:
        result.actions.append("run template: create")
        if dry_run:
            result.actions.append("scheduled run: create")
            return result
        template = client.create_run_template(definition.run_template_body(result.agent_id))
    else:
        result.actions.append("run template: exists")
    template_id = str(template["id"])

    if client.find_scheduled_run(template_id) is None:
        result.actions.append("scheduled run: create")
        if not dry_run:
            client.create_scheduled_run(scheduled_run_body(template_id))
    else:
        result.actions.append("scheduled run: exists")
    return result


@click.command(name="sync-coval")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Report what would change; write nothing."
)
@click.option("--test-set-id", default=None, help="Override coval_s2s_dental_test_set_id.")
@click.option(
    "--coval-api-base", envvar="COVAL_API_BASE", default=COVAL_API_BASE, show_default=True
)
def sync_coval(dry_run: bool, test_set_id: str | None, coval_api_base: str) -> None:
    """Create or reconcile the Coval agent, test-set link, run template, and daily schedule
    for the Phonely text agent. Laptop tool; prints the agent id the fetch job needs."""
    try:
        definition = CovalTextAgentDefinition.from_settings(get_settings(), test_set_id=test_set_id)
        if dry_run:
            click.echo(json.dumps(definition.redacted_body(), indent=2, sort_keys=True))
        with CovalTextClient(COVAL_API_KEY.resolve(), coval_api_base) as client:
            result = sync(client, definition, dry_run=dry_run)
    except (SyncError, RuntimeError, httpx.HTTPError) as exc:
        raise click.ClickException(str(exc)) from exc
    for action in result.actions:
        click.echo(action)
    click.echo(f"COVAL_LLM_PHONELY_AGENT_ID={result.agent_id or '<created on apply>'}")
