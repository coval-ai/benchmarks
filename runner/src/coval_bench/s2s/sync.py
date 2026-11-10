# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Set who runs in the S2S daily templates from the registry's ``collected``."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Protocol

import click
import httpx
import psycopg
import structlog
from google.api_core.exceptions import GoogleAPIError

from coval_bench.alerts import coval_error_text, publish_coval_failure
from coval_bench.config import get_settings
from coval_bench.db.registry_store import load_models
from coval_bench.logging import configure_logging
from coval_bench.platform_assets import COVAL_API_BASE, COVAL_API_KEY, CovalClient, SyncError
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.s2s.scenario_map import daily_scenario, read_scenario_map

logger = structlog.get_logger("coval_bench.s2s.sync")


class SyncRefused(ValueError):
    """The sync would leave the daily templates in a state it must not create."""


class RunTemplateClient(Protocol):
    def get_run_template(self, template_id: str) -> dict[str, object]: ...
    def update_run_template(
        self, template_id: str, body: dict[str, object]
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class TemplateChange:
    condition: str
    template_id: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    unknown: tuple[str, ...]
    agent_ids: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


def wanted_agents(
    agents: Mapping[str, tuple[str | None, str, str]],
    collected: Collection[tuple[str, str]],
) -> dict[str, str]:
    """Agent id to its map key, for each defined agent whose registry row is collected.

    ``agents`` maps a map key to (provider, registry model, Coval agent id). Refuses an
    empty result, which would leave the daily templates with nobody to run.
    """
    wanted: dict[str, str] = {}
    for key, (provider, model, agent_id) in agents.items():
        if provider is None:
            logger.warning("s2s_sync_agent_without_provider", agent=key)
        elif (provider, model) in collected:
            wanted[agent_id] = key
    if not wanted:
        raise SyncRefused("no collected agent is defined for the daily scenario; not emptying it")
    return wanted


def _live_agent_ids(template: dict[str, object], template_id: str) -> set[str]:
    agent_ids = template.get("agent_ids")
    if not isinstance(agent_ids, list):
        raise SyncRefused(f"run template {template_id} has no agent_ids list")
    return {str(agent_id) for agent_id in agent_ids}


def plan_templates(
    client: RunTemplateClient,
    template_ids: Mapping[str, str],
    wanted: Mapping[str, str],
    names: Mapping[str, str],
) -> list[TemplateChange]:
    """What each template's agent list should become, read before anything is written.

    Only agents the map defines are added or removed; an agent the map doesn't know is
    left in place and reported, so a hand-added experiment is never dropped.
    """
    if not template_ids:
        raise SyncRefused("the daily scenario's live version has no run templates")
    changes = []
    for condition, template_id in sorted(template_ids.items()):
        live = _live_agent_ids(client.get_run_template(template_id), template_id)
        defined = live & names.keys()
        unknown = live - names.keys()
        changes.append(
            TemplateChange(
                condition=condition,
                template_id=template_id,
                added=tuple(sorted(wanted[a] for a in wanted.keys() - live)),
                removed=tuple(sorted(names[a] for a in defined - wanted.keys())),
                unknown=tuple(sorted(unknown)),
                agent_ids=tuple(sorted(wanted.keys() | unknown)),
            )
        )
    return changes


def _coval_failure(exc: Exception) -> str:
    if isinstance(exc, SyncError):
        return str(exc).split(":", 1)[0]
    if isinstance(exc, httpx.HTTPError):
        return coval_error_text(exc)
    return type(exc).__name__


@click.command(name="sync-s2s")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Report what would change; write nothing."
)
@click.option(
    "--coval-api-base", envvar="COVAL_API_BASE", default=COVAL_API_BASE, show_default=True
)
def sync_s2s(dry_run: bool, coval_api_base: str) -> None:
    """Set who runs in the S2S daily templates from the registry's ``collected``.

    Only the daily scenario's live templates are touched; their agents come from the
    scenario map, kept when their registry row is collected.
    """
    settings = get_settings()
    configure_logging(level=settings.log_level)
    try:
        scenario = daily_scenario(read_scenario_map(settings.s2s_scenario_map_uri.strip()))
        agents = {
            key: (agent.provider, agent.registry_model, agent.id)
            for key, agent in scenario.agents.items()
        }
        collected = frozenset(
            (m.provider, m.model)
            for m in load_models(settings)
            if m.benchmark is Benchmark.S2S and m.collected
        )
        wanted = wanted_agents(agents, collected)
        names = {agent_id: key for key, (_, _, agent_id) in agents.items()}
    except (SyncRefused, ValueError, RuntimeError, psycopg.Error, GoogleAPIError) as exc:
        logger.error("RUN_FAILED", error=str(exc))
        raise click.ClickException(str(exc)) from exc

    applied: list[str] = []
    try:
        with CovalClient(COVAL_API_KEY.resolve(), coval_api_base) as client:
            changes = plan_templates(client, scenario.run_template_ids, wanted, names)
            for change in changes:
                if change.changed and not dry_run:
                    client.update_run_template(
                        change.template_id, {"agent_ids": list(change.agent_ids)}
                    )
                    applied.append(change.condition)
    except SyncRefused as exc:
        logger.error("RUN_FAILED", error=str(exc))
        raise click.ClickException(str(exc)) from exc
    except (SyncError, httpx.HTTPError) as exc:
        error = _coval_failure(exc)
        logger.error("RUN_FAILED", error=error, applied=applied)
        if not dry_run:
            publish_coval_failure(
                settings.alerts_topic,
                job="s2s-sync",
                failure="template_sync",
                error=f"{error}; updated before failing: {', '.join(applied) or 'none'}",
            )
        raise click.ClickException(error) from exc

    for change in changes:
        if change.unknown:
            logger.warning(
                "s2s_sync_unknown_agents",
                condition=change.condition,
                template_id=change.template_id,
                agent_ids=list(change.unknown),
            )
        logger.info(
            "s2s_sync_template",
            condition=change.condition,
            template_id=change.template_id,
            added=list(change.added),
            removed=list(change.removed),
            dry_run=dry_run,
        )
        summary = (
            f"+ {', '.join(change.added) or '-'}  − {', '.join(change.removed) or '-'}"
            if change.changed
            else "unchanged"
        )
        click.echo(f"{change.condition} ({change.template_id}): {summary}")
