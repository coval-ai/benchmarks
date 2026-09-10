# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The Coval side of the Phonely text agent, defined in code and pushed idempotently."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Self

import click
import httpx
import psycopg
import structlog
from pydantic import BaseModel, SecretStr

from coval_bench.config import Settings, get_settings
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.registry_store import fetch_models
from coval_bench.llm import benchmark
from coval_bench.logging import configure_logging
from coval_bench.platform_assets import COVAL_API_BASE, COVAL_API_KEY, CovalClient, SyncError, plan
from coval_bench.registries.models import RegisteredModel
from coval_bench.variants.platforms import redact

# The public API rejects MODEL_TYPE_TEXT; CHAT is the HTTP text simulator.
MODEL_TYPE = "MODEL_TYPE_CHAT"
SCHEDULE_EXPRESSION = "cron(0 13 * * ? *)"
SCHEDULE_TIMEZONE = "UTC"
INPUT_TEMPLATE = (
    '{"model": "{{session_id}}", "messages": {{messages}}, '
    '"simulation_id": "{{simulation_output_id}}"}'
)
# PATCH replaces metadata wholesale, so it is reconciled as one value.
MANAGED = ("display_name", "metadata", "test_set_ids")


class CovalTextAgentDefinition(BaseModel, frozen=True):
    provider: str
    proxy_url: str
    proxy_secret: SecretStr
    test_set_id: str
    instruction_metric_id: str
    collected: bool = True

    @property
    def customer_agent_id(self) -> str:
        return f"benchmarks-{self.provider}-text"

    @property
    def display_name(self) -> str:
        return f"Benchmarks: {self.provider.capitalize()} text agent"

    @property
    def run_name(self) -> str:
        return f"benchmarks-{self.provider}-text-daily"

    @classmethod
    def from_settings(
        cls,
        provider: str,
        settings: Settings,
        *,
        test_set_id: str | None = None,
        collected: bool = True,
    ) -> Self:
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
            raise SyncError(f"sync-llm needs {', '.join(missing)} set")
        return cls(
            provider=provider,
            proxy_url=proxy_url.rstrip("/"),
            proxy_secret=proxy_secret,
            test_set_id=dental,
            instruction_metric_id=metric,
            collected=collected,
        )

    def agent_body(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "customer_agent_id": self.customer_agent_id,
            "model_type": MODEL_TYPE,
            "metadata": {
                "chat_endpoint": f"{self.proxy_url}/llm/{self.provider}/chat",
                "initialization_endpoint": f"{self.proxy_url}/llm/{self.provider}/session",
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
        return benchmark.run_template_body(
            self.run_name, agent_id, self.test_set_id, self.instruction_metric_id
        )


def scheduled_run_body(run_name: str, run_template_id: str, *, enabled: bool) -> dict[str, Any]:
    return {
        "display_name": run_name,
        "run_template_id": run_template_id,
        "schedule_expression": SCHEDULE_EXPRESSION,
        "schedule_timezone": SCHEDULE_TIMEZONE,
        "enabled": enabled,
    }


def load_llm_models(settings: Settings) -> list[RegisteredModel]:
    async def _load() -> list[RegisteredModel]:
        async with lifespan_pool(settings) as pool:
            return benchmark.llm_models(await fetch_models(pool))

    return asyncio.run(_load())


class CovalTextClient(CovalClient):
    def __enter__(self) -> Self:
        return self

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

    def update_run_template(self, template_id: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("PATCH", f"/run-templates/{template_id}", body)
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

    def update_scheduled_run(self, scheduled_id: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("PATCH", f"/scheduled-runs/{scheduled_id}", body)
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
    live = client.find_agent(definition.customer_agent_id)
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

    template = client.find_run_template(definition.run_name)
    if template is None:
        result.actions.append("run template: create")
        if dry_run:
            result.actions.append("scheduled run: create")
            return result
        template = client.create_run_template(definition.run_template_body(result.agent_id))
    else:
        wanted_template = definition.run_template_body(result.agent_id)
        drift = plan(template, {path: wanted_template[path] for path in benchmark.TEMPLATE_MANAGED})
        if drift.update:
            result.actions.append(f"run template: patch {sorted(drift.update)}")
            if not dry_run:
                client.update_run_template(
                    str(template["id"]),
                    benchmark.template_patch_body(wanted_template, drift.update),
                )
        else:
            result.actions.append("run template: unchanged")
    template_id = str(template["id"])

    scheduled = client.find_scheduled_run(template_id)
    if scheduled is None:
        result.actions.append("scheduled run: create")
        if not dry_run:
            client.create_scheduled_run(
                scheduled_run_body(definition.run_name, template_id, enabled=definition.collected)
            )
    elif bool(scheduled.get("enabled")) != definition.collected:
        result.actions.append(f"scheduled run: {'enable' if definition.collected else 'disable'}")
        if not dry_run:
            client.update_scheduled_run(str(scheduled["id"]), {"enabled": definition.collected})
    else:
        result.actions.append("scheduled run: unchanged")
    return result


@click.command(name="sync-llm")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Report what would change; write nothing."
)
@click.option("--test-set-id", default=None, help="Override coval_s2s_dental_test_set_id.")
@click.option(
    "--coval-api-base", envvar="COVAL_API_BASE", default=COVAL_API_BASE, show_default=True
)
def sync_llm(dry_run: bool, test_set_id: str | None, coval_api_base: str) -> None:
    """Reconcile the Coval text benchmark, then ingest its completed runs.

    The first apply creates the scheduled benchmark and defers ingestion until a
    run can exist. Later applies sync both Coval's desired state and our database.
    """
    settings = get_settings()
    if not dry_run:
        configure_logging(level=settings.log_level)
    run_logger = structlog.get_logger("coval_bench.llm.coval_agent")
    try:
        definitions = [
            CovalTextAgentDefinition.from_settings(
                model.provider, settings, test_set_id=test_set_id, collected=model.collected
            )
            for model in load_llm_models(settings)
        ]
        if dry_run:
            for definition in definitions:
                click.echo(json.dumps(definition.redacted_body(), indent=2, sort_keys=True))
        with CovalTextClient(COVAL_API_KEY.resolve(), coval_api_base) as client:
            results = {
                definition.provider: sync(client, definition, dry_run=dry_run)
                for definition in definitions
            }
    except (SyncError, RuntimeError, httpx.HTTPError, psycopg.Error) as exc:
        if not dry_run:
            run_logger.error("RUN_FAILED", error=str(exc), exc_info=exc)
        raise click.ClickException(str(exc)) from exc
    for provider, result in results.items():
        for action in result.actions:
            click.echo(f"{provider} {action}")
        click.echo(
            f"COVAL_LLM_{provider.upper()}_AGENT_ID={result.agent_id or '<created on apply>'}"
        )
    if dry_run:
        return
    fetchable: dict[str, str] = {}
    for definition, result in zip(definitions, results.values(), strict=True):
        provider = definition.provider
        if not definition.collected:
            run_logger.info("llm_sync_fetch_skipped", provider=provider, reason="not_collected")
        elif "scheduled run: create" in result.actions:
            run_logger.info(
                "llm_sync_fetch_deferred", provider=provider, reason="scheduled_run_created"
            )
        else:
            fetchable[provider] = result.agent_id
    if fetchable:
        from coval_bench.registries.benchmarks import Benchmark
        from coval_bench.s2s.fetch_v2v import _run_fetch

        _run_fetch(Benchmark.LLM, (), 720, 100, settings=settings, llm_agent_ids=fetchable)
    for provider, result in results.items():
        run_logger.info(
            "llm_sync_completed",
            provider=provider,
            agent_id=result.agent_id,
            actions=result.actions,
        )
