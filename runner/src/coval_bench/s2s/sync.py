# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Set who runs and what is scored in the S2S daily templates from ``collected``."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

import click
import httpx
import psycopg
import structlog
from google.api_core.exceptions import GoogleAPIError

from coval_bench.alerts import coval_error_text, publish_coval_failure
from coval_bench.config import get_settings
from coval_bench.db.metric_store import load_collected_metrics
from coval_bench.db.registry_store import load_models
from coval_bench.logging import configure_logging
from coval_bench.platform_assets import COVAL_API_BASE, COVAL_API_KEY, CovalClient, SyncError
from coval_bench.registries import Metric
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.s2s.scenario_map import (
    DAILY_ANCHOR,
    daily_metrics,
    daily_scenario,
    read_scenario_map,
)

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
    metrics_added: tuple[str, ...] = ()
    metrics_removed: tuple[str, ...] = ()
    unknown_metrics: tuple[str, ...] = ()
    metric_ids: tuple[str, ...] = ()

    @property
    def body(self) -> dict[str, object]:
        """The fields to send to Coval; empty when the template already matches."""
        body: dict[str, object] = {}
        if self.added or self.removed:
            body["agent_ids"] = list(self.agent_ids)
        if self.metrics_added or self.metrics_removed:
            body["metric_ids"] = list(self.metric_ids)
        return body

    @property
    def changed(self) -> bool:
        return bool(self.body)


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


def wanted_metrics(
    metrics: Mapping[str, tuple[Metric, str]],
    collected: Collection[str],
    anchor: Metric,
) -> dict[str, str]:
    """Coval metric id to its map name, for each allowed metric that is collected.

    ``metrics`` maps a map name to (our metric, Coval metric id). Refuses to drop the
    anchor: the fetch writes nothing for a run without it.
    """
    if anchor not in {metric for metric, _ in metrics.values()}:
        raise SyncRefused(f"the daily scenario does not list its anchor metric {anchor.value}")
    if anchor.value not in collected:
        raise SyncRefused(f"{anchor.value} anchors the daily datasets; not removing it")
    return {
        coval_id: name for name, (metric, coval_id) in metrics.items() if metric.value in collected
    }


def _live_ids(template: dict[str, object], template_id: str, field: str) -> set[str]:
    ids = template.get(field)
    if not isinstance(ids, list):
        raise SyncRefused(f"run template {template_id} has no {field} list")
    return {str(item) for item in ids}


def plan_templates(
    client: RunTemplateClient,
    template_ids: Mapping[str, str],
    wanted: Mapping[str, str],
    names: Mapping[str, str],
    metrics: Mapping[str, str] | None = None,
    metric_names: Mapping[str, str] | None = None,
) -> list[TemplateChange]:
    """What each template's agent and metric lists should become, read before any write.

    Only agents and metrics the map defines are added or removed; one the map doesn't
    know is left in place and reported, so a hand-added experiment is never dropped.
    ``metrics`` is the wanted metrics by Coval id; None leaves every metric list alone.
    """
    if not template_ids:
        raise SyncRefused("the daily scenario's live version has no run templates")
    changes = []
    for condition, template_id in sorted(template_ids.items()):
        template = client.get_run_template(template_id)
        live = _live_ids(template, template_id, "agent_ids")
        defined = live & names.keys()
        unknown = live - names.keys()
        change = TemplateChange(
            condition=condition,
            template_id=template_id,
            added=tuple(sorted(wanted[a] for a in wanted.keys() - live)),
            removed=tuple(sorted(names[a] for a in defined - wanted.keys())),
            unknown=tuple(sorted(unknown)),
            agent_ids=tuple(sorted(wanted.keys() | unknown)),
        )
        if metrics is not None and metric_names is not None:
            scored = _live_ids(template, template_id, "metric_ids")
            stray = scored - metric_names.keys()
            change = replace(
                change,
                metrics_added=tuple(sorted(metrics[m] for m in metrics.keys() - scored)),
                metrics_removed=tuple(
                    sorted(metric_names[m] for m in (scored & metric_names.keys()) - metrics.keys())
                ),
                unknown_metrics=tuple(sorted(stray)),
                metric_ids=tuple(sorted(metrics.keys() | stray)),
            )
        changes.append(change)
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
    """Set who runs and what is scored in the S2S daily templates from ``collected``.

    Only the daily scenario's live templates are touched; their agents and metrics come
    from the scenario map, kept when their registry row or metric is collected.
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
        allowed = daily_metrics(scenario)
        metric_names = {coval_id: name for name, (_, coval_id) in allowed.items()}
        metrics: dict[str, str] | None
        try:
            metrics = wanted_metrics(allowed, load_collected_metrics(settings), DAILY_ANCHOR)
        except psycopg.errors.UndefinedColumn:
            logger.warning("s2s_sync_metric_flags_missing")
            metrics = None
        except SyncRefused as exc:
            logger.error("s2s_sync_metrics_refused", error=str(exc))
            publish_coval_failure(
                settings.alerts_topic, job="s2s-sync", failure="metric_sync", error=str(exc)
            )
            metrics = None
    except (SyncRefused, ValueError, RuntimeError, psycopg.Error, GoogleAPIError) as exc:
        logger.error("RUN_FAILED", error=str(exc))
        raise click.ClickException(str(exc)) from exc

    applied: list[str] = []
    try:
        with CovalClient(COVAL_API_KEY.resolve(), coval_api_base) as client:
            changes = plan_templates(
                client, scenario.run_template_ids, wanted, names, metrics, metric_names
            )
            for change in changes:
                if change.changed and not dry_run:
                    client.update_run_template(change.template_id, change.body)
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
        if change.unknown_metrics:
            logger.warning(
                "s2s_sync_unknown_metrics",
                condition=change.condition,
                template_id=change.template_id,
                metric_ids=list(change.unknown_metrics),
            )
        logger.info(
            "s2s_sync_template",
            condition=change.condition,
            template_id=change.template_id,
            added=list(change.added),
            removed=list(change.removed),
            metrics_added=list(change.metrics_added),
            metrics_removed=list(change.metrics_removed),
            dry_run=dry_run,
        )
        added = (*change.added, *change.metrics_added)
        removed = (*change.removed, *change.metrics_removed)
        summary = (
            f"+ {', '.join(added) or '-'}  − {', '.join(removed) or '-'}"
            if change.changed
            else "unchanged"
        )
        click.echo(f"{change.condition} ({change.template_id}): {summary}")
