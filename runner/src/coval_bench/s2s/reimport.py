# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace stored S2S runs with a fresh read of the same Coval runs."""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import click
import httpx
import psycopg
import psycopg.rows
from google.cloud import storage
from psycopg_pool import AsyncConnectionPool

from coval_bench.config import Settings, get_settings, require_capture_bucket
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.dashboard_rollups import ENQUEUE_SLOT_SQL, RUN_SLOT, fill_rollup
from coval_bench.db.models import RunStatus
from coval_bench.db.registry_store import fetch_models
from coval_bench.db.writer import RunWriter
from coval_bench.logging import configure_logging
from coval_bench.registries import Metric
from coval_bench.runner.capture import preflight_capture_storage, read_import_run_claim
from coval_bench.s2s.conditions import (
    PLATFORM_SLUGS,
    ROOMS,
    Condition,
    DatasetMetrics,
    condition_for,
    room_dataset_id,
    scenario_family,
)
from coval_bench.s2s.coval_runs import CovalRun, coval_client, recent_completed_runs
from coval_bench.s2s.identity import (
    all_persona_conditions,
    dataset_identity,
    parse_persona_conditions,
)
from coval_bench.s2s.ingest import import_identity, ingest_run, resolve_import_generation
from coval_bench.s2s.rows import ingestable_metrics, metric_values
from coval_bench.s2s.scenario_map import with_scenario_map
from coval_bench.s2s.specs import AgentSpec, platform_specs

Counts = dict[tuple[str, str], int]

_STORED_SQL = """
SELECT o.run_id, m.code AS metric, e.status, count(*) AS n
FROM benchmarks_v2.benchmark_observations o
JOIN benchmarks_v2.runs r ON r.id = o.run_id
JOIN benchmarks_v2.metric_evaluations e ON e.observation_id = o.id
JOIN benchmarks_v2.metrics m ON m.id = e.metric_id
WHERE o.provider = %(provider)s
  AND o.benchmark = 'S2S'
  AND split_part(o.sample_id, '/', 1) = %(coval_run_id)s
  AND r.status IN ('succeeded', 'partial')
GROUP BY o.run_id, m.code, e.status
"""

_CALLS_SQL = """
SELECT o.run_id, count(*) AS n
FROM benchmarks_v2.benchmark_observations o
JOIN benchmarks_v2.runs r ON r.id = o.run_id
WHERE o.provider = %(provider)s
  AND o.benchmark = 'S2S'
  AND split_part(o.sample_id, '/', 1) = %(coval_run_id)s
  AND r.status IN ('succeeded', 'partial')
GROUP BY o.run_id
"""


class SwapRefused(RuntimeError):
    """The stored rows are not what the swap was planned against."""


@dataclass(frozen=True)
class StoredRun:
    run_id: int
    calls: int
    counts: Counts

    @property
    def metrics(self) -> frozenset[str]:
        return frozenset(metric for metric, _ in self.counts)


@dataclass(frozen=True)
class Candidate:
    spec: AgentSpec
    coval_run: CovalRun
    dataset_id: str
    dataset_sha256: str
    condition: DatasetMetrics


async def _stored(
    cur: psycopg.AsyncCursor[psycopg.rows.DictRow], provider: str, coval_run_id: str
) -> dict[int, StoredRun]:
    params = {"provider": provider, "coval_run_id": coval_run_id}
    await cur.execute(_STORED_SQL, params)
    counts: dict[int, Counts] = {}
    for row in await cur.fetchall():
        counts.setdefault(int(row["run_id"]), {})[(str(row["metric"]), str(row["status"]))] = int(
            row["n"]
        )
    await cur.execute(_CALLS_SQL, params)
    calls = {int(row["run_id"]): int(row["n"]) for row in await cur.fetchall()}
    return {
        run_id: StoredRun(run_id=run_id, calls=calls.get(run_id, 0), counts=counts.get(run_id, {}))
        for run_id in calls.keys() | counts.keys()
    }


async def stored_runs(
    pool: AsyncConnectionPool[Any], provider: str, coval_run_id: str
) -> dict[int, StoredRun]:
    """Every counted run holding rows for this Coval run, with its rows per metric and status."""
    async with pool.connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        return await _stored(cur, provider, coval_run_id)


async def swap_runs(
    pool: AsyncConnectionPool[Any],
    *,
    provider: str,
    coval_run_id: str,
    keep: int,
    replace: Mapping[int, StoredRun],
) -> None:
    """Delete the replaced runs in one transaction, or change nothing.

    The replaced runs must still hold exactly the rows they were planned with, and
    afterwards only ``keep`` may hold rows for the Coval run.
    """
    old = sorted(replace)
    if not old or keep in replace:
        raise SwapRefused("nothing to replace, or the kept run is among the replaced")
    async with (
        pool.connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
    ):
        await cur.execute(
            """SELECT id, scheduled_at FROM benchmarks_v2.runs
               WHERE id = ANY(%(ids)s) ORDER BY id FOR UPDATE""",
            {"ids": [*old, keep]},
        )
        slots = {row["scheduled_at"] for row in await cur.fetchall() if row["scheduled_at"]}
        before = await _stored(cur, provider, coval_run_id)
        if {run_id: before.get(run_id) for run_id in old} != dict(replace):
            raise SwapRefused(f"stored rows for {coval_run_id} changed since they were counted")
        if keep not in before:
            raise SwapRefused(f"run {keep} holds no counted rows for {coval_run_id}")
        await cur.execute("DELETE FROM benchmarks_v2.runs WHERE id = ANY(%(ids)s)", {"ids": old})
        if cur.rowcount != len(old):
            raise SwapRefused(f"deleted {cur.rowcount} runs, expected {len(old)}")
        after = await _stored(cur, provider, coval_run_id)
        if after != {keep: before[keep]}:
            raise SwapRefused(f"after the swap {coval_run_id} is held by runs {sorted(after)}")
        await cur.execute(ENQUEUE_SLOT_SQL, {"run_id": keep})
        for slot in sorted(slots):
            await fill_rollup(conn, grain=RUN_SLOT, bucket_at=slot)


def _days(since: datetime, until: datetime | None) -> tuple[datetime, datetime]:
    start = since.replace(tzinfo=UTC)
    end = (until.replace(tzinfo=UTC) if until else datetime.now(tz=UTC)) + timedelta(days=1)
    return start, end


async def find_candidates(
    client: httpx.AsyncClient,
    settings: Settings,
    specs: Collection[AgentSpec],
    *,
    since: datetime,
    until: datetime | None,
    room_dataset: str | None,
    coval_run_ids: frozenset[str],
    page_size: int = 100,
) -> list[Candidate]:
    """The completed Coval runs of these agents in the window, with their dataset."""
    start, end = _days(since, until)
    persona_conditions = parse_persona_conditions(all_persona_conditions(settings))
    candidates: list[Candidate] = []
    for spec in specs:
        if not spec.agent_id:
            continue
        runs = await recent_completed_runs(
            client,
            spec.agent_id,
            period_seconds=settings.s2s_fetch_period_seconds,
            test_set_id=spec.test_set_id,
            window_seconds=int((datetime.now(tz=UTC) - start).total_seconds()) + 1,
            page_size=page_size,
            workspace_id=spec.workspace_id,
        )
        for run in runs:
            if run.create_time is None or not start <= run.create_time < end:
                continue
            if coval_run_ids and run.run_id not in coval_run_ids:
                continue
            identity = dataset_identity(
                spec.test_set_id,
                run.persona_id,
                family=spec.family,
                persona_conditions=persona_conditions,
            )
            if identity is None:
                continue
            dataset_id, dataset_sha256 = identity
            if room_dataset is not None and dataset_id != room_dataset:
                continue
            candidates.append(
                Candidate(
                    spec=spec,
                    coval_run=run,
                    dataset_id=dataset_id,
                    dataset_sha256=dataset_sha256,
                    condition=condition_for(dataset_id),
                )
            )
    return sorted(candidates, key=lambda c: (c.spec.provider, c.coval_run.create_time or start))


async def coval_calls(
    client: httpx.AsyncClient, candidate: Candidate, named: frozenset[str]
) -> int | None:
    """How many calls a fresh import would store; None when the anchor metric is absent.

    The anchor's calls, plus the calls the optional latency adds as failed rows.
    """
    headers = (
        {"X-Coval-Workspace-Id": candidate.spec.workspace_id}
        if candidate.spec.workspace_id
        else None
    )
    resp = await client.get(f"/runs/{candidate.coval_run.run_id}", headers=headers)
    resp.raise_for_status()
    run = resp.json()["run"]
    results = run.get("results") or {}
    anchor = candidate.condition.required
    values = metric_values(results.get("metrics") or {}, candidate.spec.metric_ids.get(anchor))
    if values is None:
        return None
    calls = {str(v.get("simulation_output_id") or f"#{i}") for i, v in enumerate(values)}
    if anchor is not Metric.V2V and str(Metric.V2V) in named:
        calls |= {s for s in results.get("output_ids") or [] if isinstance(s, str)}
    return len(calls)


def decide(
    stored: Mapping[int, StoredRun], named: frozenset[str], calls_now: int | None
) -> tuple[str, str]:
    """(action, reason): replace, finish (swap only), unchanged, or refuse."""
    if not stored:
        return "refuse", "nothing stored; the normal fetch imports it"
    if calls_now is None:
        return "refuse", "the anchor metric is not on the Coval run"
    held = frozenset().union(*(run.metrics for run in stored.values()))
    if held != named:
        return "refuse", f"stored metrics {sorted(held)} differ from --metric {sorted(named)}"
    newest = stored[max(stored)]
    if newest.calls == calls_now and newest.metrics == named:
        return (
            ("unchanged", "already matches Coval")
            if len(stored) == 1
            else (
                "finish",
                "an earlier --apply wrote the new run but did not swap",
            )
        )
    return "replace", f"{newest.calls} calls stored, {calls_now} on Coval"


def _describe(stored: Mapping[int, StoredRun]) -> str:
    parts = []
    for run_id, run in sorted(stored.items()):
        rows = ", ".join(f"{m}:{s}={n}" for (m, s), n in sorted(run.counts.items()))
        parts.append(f"run {run_id} calls={run.calls} [{rows}]")
    return "; ".join(parts)


async def _apply(
    client: httpx.AsyncClient,
    pool: AsyncConnectionPool[Any],
    writer: RunWriter,
    settings: Settings,
    candidate: Candidate,
    *,
    action: str,
    stored: Mapping[int, StoredRun],
    named: frozenset[str],
    artifact_client: storage.Client,
) -> str:
    spec, coval_run = candidate.spec, candidate.coval_run
    if action == "finish":
        keep = max(stored)
        await swap_runs(
            pool,
            provider=spec.provider,
            coval_run_id=coval_run.run_id,
            keep=keep,
            replace={run_id: run for run_id, run in stored.items() if run_id != keep},
        )
        return f"swapped in run {keep}"
    bucket = settings.benchmark_artifact_bucket
    identity = import_identity(
        spec=spec,
        coval_run=coval_run,
        dataset_id=candidate.dataset_id,
        dataset_sha256=candidate.dataset_sha256,
        workspace_id=spec.workspace_id,
    )
    claim, generation = await resolve_import_generation(artifact_client, bucket, identity)
    if claim is not None or generation is None:
        return "refused: another import of this run is in progress or owes recovery"
    status = await ingest_run(
        client,
        writer,
        spec=spec,
        coval_run=coval_run,
        metric_ids=spec.metric_ids,
        condition=candidate.condition,
        pending=frozenset(Metric(m) for m in named),
        dataset_id=candidate.dataset_id,
        dataset_sha256=candidate.dataset_sha256,
        period_seconds=settings.s2s_fetch_period_seconds,
        artifact_client=artifact_client,
        artifact_bucket=bucket,
        workspace_id=spec.workspace_id,
        import_identity=identity,
        import_generation=generation,
        rebuild_rollup=False,
    )
    if status is None or status is RunStatus.FAILED:
        return f"refused: the new import came back {status}; nothing replaced"
    new_claim = await asyncio.to_thread(
        read_import_run_claim, artifact_client, bucket, identity, generation
    )
    if new_claim is None:
        return "refused: the new import left no claim; nothing replaced"
    new_run = await writer.get_run(new_claim.run_id)
    if new_run.error:
        return (
            f"refused: new run {new_claim.run_id} is incomplete ({new_run.error}); nothing replaced"
        )
    await swap_runs(
        pool,
        provider=spec.provider,
        coval_run_id=coval_run.run_id,
        keep=new_claim.run_id,
        replace=stored,
    )
    return f"replaced by run {new_claim.run_id} ({status})"


async def reimport(
    *,
    scenario: str,
    since: datetime,
    until: datetime | None,
    provider: str | None,
    room: Condition | None,
    coval_run_ids: frozenset[str],
    metrics: frozenset[str],
    apply: bool,
    settings: Settings,
) -> int:
    """Print one line per Coval run; with ``apply``, replace the ones that need it.

    Returns how many runs were refused or failed.
    """
    problems = 0
    async with coval_client(settings) as client, lifespan_pool(settings) as pool:
        models = await fetch_models(pool)
        family = scenario_family(scenario)
        specs = [
            spec
            for spec in platform_specs(settings, models)
            if spec.family == family and (provider is None or spec.provider == provider)
        ]
        writer = RunWriter(pool)
        artifact_client = storage.Client() if apply else None
        if artifact_client is not None:
            await asyncio.to_thread(
                preflight_capture_storage, artifact_client, settings.benchmark_artifact_bucket
            )
            await writer.preflight_required_capture_schema()
        candidates = await find_candidates(
            client,
            settings,
            specs,
            since=since,
            until=until,
            room_dataset=room_dataset_id(scenario, room) if room else None,
            coval_run_ids=coval_run_ids,
        )
        for candidate in candidates:
            run = candidate.coval_run
            created = run.create_time.strftime("%Y-%m-%d %H:%M") if run.create_time else "?"
            label = f"{candidate.spec.provider} {candidate.dataset_id} {run.run_id} {created}"
            fetched = ingestable_metrics(candidate.condition, candidate.spec.metric_ids)
            unknown = metrics - {str(metric) for metric in fetched}
            if unknown:
                click.echo(f"{label}: refuse: {sorted(unknown)} are not fetched for this dataset")
                problems += 1
                continue
            stored = await stored_runs(pool, candidate.spec.provider, run.run_id)
            action, reason = decide(stored, metrics, await coval_calls(client, candidate, metrics))
            click.echo(f"{label}: {action} ({reason}) | {_describe(stored)}")
            if action == "refuse":
                problems += 1
            if artifact_client is None or action not in ("replace", "finish"):
                continue
            try:
                outcome = await _apply(
                    client,
                    pool,
                    writer,
                    settings,
                    candidate,
                    action=action,
                    stored=stored,
                    named=metrics,
                    artifact_client=artifact_client,
                )
            except SwapRefused as exc:
                outcome = f"refused: {exc}"
            if outcome.startswith("refused"):
                problems += 1
            after = await stored_runs(pool, candidate.spec.provider, run.run_id)
            click.echo(f"  -> {outcome} | {_describe(after)}")
        if apply:
            await writer.refresh_window_views()
        click.echo(f"{len(candidates)} runs, {problems} refused or failed")
    return problems


@click.command(name="reimport-s2s")
@click.option("--scenario", required=True, type=click.Choice(PLATFORM_SLUGS))
@click.option("--since", required=True, type=click.DateTime(["%Y-%m-%d"]), help="First day (UTC).")
@click.option("--until", type=click.DateTime(["%Y-%m-%d"]), help="Last day (UTC), inclusive.")
@click.option("--provider")
@click.option("--room", type=click.Choice([room.value for room in ROOMS]))
@click.option("--coval-run-id", "coval_run_ids", multiple=True)
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    required=True,
    type=click.Choice([metric.value for metric in Metric]),
    help="Every metric the runs store; a run storing any other set is refused. Repeatable.",
)
@click.option("--apply", is_flag=True, default=False, help="Write; without it this is a dry run.")
def reimport_s2s(
    scenario: str,
    since: datetime,
    until: datetime | None,
    provider: str | None,
    room: str | None,
    coval_run_ids: tuple[str, ...],
    metrics: tuple[str, ...],
    apply: bool,
) -> None:
    """Replace stored S2S runs with a fresh read of the same Coval runs."""
    settings = with_scenario_map(get_settings())
    configure_logging(level=settings.log_level)
    if apply:
        require_capture_bucket(settings)
    problems = asyncio.run(
        reimport(
            scenario=scenario,
            since=since,
            until=until,
            provider=provider,
            room=Condition(room) if room else None,
            coval_run_ids=frozenset(coval_run_ids),
            metrics=frozenset(metrics),
            apply=apply,
            settings=settings,
        )
    )
    if problems:
        raise click.ClickException(f"{problems} runs refused or failed")
