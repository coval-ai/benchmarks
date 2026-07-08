# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch S2S (voice-to-voice) latency from the Coval API and write per-clip rows.

For each provider, find its newest completed Coval run, read the per-clip
latency values (seconds), convert to milliseconds, and write one results row
per clip via ``RunWriter``; the existing matviews aggregate. Agent ids, the
metric id, and the API key are read from the environment, never committed.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.resources
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import click
import httpx
import structlog

from coval_bench.config import Settings, get_settings
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.models import Result, ResultStatus, RunStatus
from coval_bench.db.writer import RunWriter
from coval_bench.registries import Metric
from coval_bench.registries.benchmarks import Benchmark

logger = structlog.get_logger("coval_bench.s2s.fetch_v2v")

# Daily job: floor scheduled_at to the day so a run's clip rows share one bucket.
DAILY_PERIOD_SECONDS = 86_400

# The dataset the S2S sims run against (matches the packaged manifest name).
DATASET_ID = "s2s-v1"


@dataclass(frozen=True)
class AgentSpec:
    """One S2S provider: the Settings attr holding its Coval agent id + display strings."""

    agent_id_attr: str
    provider: str
    model: str


# Agent ids resolved from Settings; model strings are display labels.
AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec(agent_id_attr="coval_s2s_openai_agent_id", provider="openai", model="gpt-realtime"),
    AgentSpec(agent_id_attr="coval_s2s_gemini_agent_id", provider="google", model="gemini-live"),
)


def _client(settings: Settings) -> httpx.AsyncClient:
    """Build the Coval API client. The key is used only as a header, never logged."""
    key = settings.coval_api_key
    if key is None:
        raise RuntimeError("coval_api_key is not set (Secret Manager in prod, .env locally)")
    return httpx.AsyncClient(
        base_url=settings.coval_api_base,
        headers={"X-API-Key": key.get_secret_value()},
        timeout=30.0,
    )


def _dataset_sha256() -> str:
    """SHA-256 of the packaged S2S manifest."""
    try:
        ref = importlib.resources.files("coval_bench.datasets.manifests").joinpath(
            f"{DATASET_ID}.json"
        )
        return hashlib.sha256(ref.read_bytes()).hexdigest()
    except Exception:
        logger.warning("dataset_sha256_failed", dataset_id=DATASET_ID, exc_info=True)
        return "unknown"


async def latest_completed_run_id(client: httpx.AsyncClient, agent_id: str) -> str | None:
    """Newest completed Coval run for one agent.

    List returns the summary view newest-first; filter by agent_id + status
    (tags are not filterable).
    """
    resp = await client.get(
        "/runs",
        params={
            "filter": f'status="COMPLETED" AND agent_id="{agent_id}"',
            "order_by": "-create_time",
            "page_size": 1,
        },
    )
    resp.raise_for_status()
    runs = cast("list[dict[str, Any]]", resp.json().get("runs", []))
    return cast("str | None", runs[0]["run_id"]) if runs else None


async def per_clip_rows(
    client: httpx.AsyncClient,
    *,
    run_pk: int,
    coval_run_id: str,
    metric_id: str,
    spec: AgentSpec,
) -> list[Result]:
    """Map a Coval run's per-clip values to Result rows.

    Values are seconds; convert to ms. A clip with no numeric value becomes a
    FAILED row, so reliability = success/total.
    """
    resp = await client.get(f"/runs/{coval_run_id}")
    resp.raise_for_status()
    run = cast("dict[str, Any]", resp.json()["run"])
    metrics = cast("dict[str, Any]", (run.get("results") or {}).get("metrics") or {})
    metric = metrics.get(metric_id)
    if metric is None:
        logger.warning(
            "metric_absent", provider=spec.provider, run_id=coval_run_id, metric_id=metric_id
        )
        return []

    values = cast("list[dict[str, Any]]", metric.get("values", []))
    rows: list[Result] = []
    for i, v in enumerate(values):
        raw = v.get("value")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            metric_value: float | None = round(float(raw) * 1000, 1)
            status = ResultStatus.SUCCESS
        else:
            metric_value = None
            status = ResultStatus.FAILED
        sim_id = v.get("simulation_output_id")
        clip_key = f"{coval_run_id}/{sim_id}" if sim_id else f"{coval_run_id}/{i}"
        rows.append(
            Result(
                run_id=run_pk,
                provider=spec.provider,
                model=spec.model,
                benchmark=Benchmark.S2S,
                metric_type=Metric.V2V,
                metric_units="milliseconds",
                metric_value=metric_value,
                audio_filename=clip_key,
                status=status,
            )
        )

    logger.info(
        "fetched_clips",
        provider=spec.provider,
        run_id=coval_run_id,
        clips=len(rows),
        success=sum(1 for r in rows if r.status is ResultStatus.SUCCESS),
    )
    return rows


async def _fetch_one_provider(
    client: httpx.AsyncClient,
    writer: RunWriter,
    *,
    spec: AgentSpec,
    agent_id: str,
    metric_id: str,
    runner_sha: str,
    scheduled_at: datetime,
) -> RunStatus:
    """Fetch one provider's latest run into its own run row; return its status.

    SUCCEEDED = all clips numeric, PARTIAL = some clips failed, FAILED = no
    clips or all failed. Errors are caught here so one provider can't abort others.
    Skips (no write) if this Coval run was already ingested, so a retry or a
    re-pulled stale run doesn't double-count and the last good data stays.
    """
    run_pk: int | None = None
    try:
        coval_run_id = await latest_completed_run_id(client, agent_id)
        if coval_run_id is None:
            logger.warning(
                "no_completed_run", provider=spec.provider, agent_attr=spec.agent_id_attr
            )
            return RunStatus.FAILED
        if await writer.coval_run_ingested(provider=spec.provider, coval_run_id=coval_run_id):
            logger.info("run_already_ingested", provider=spec.provider, coval_run_id=coval_run_id)
            return RunStatus.SUCCEEDED

        run = await writer.start_run(
            runner_sha=runner_sha,
            dataset_id=DATASET_ID,
            dataset_sha256=_dataset_sha256(),
            scheduled_at=scheduled_at,
        )
        if run.id is None:  # pragma: no cover -- start_run always returns an id
            raise RuntimeError("start_run returned a run with no id")
        run_pk = run.id
        rows = await per_clip_rows(
            client, run_pk=run_pk, coval_run_id=coval_run_id, metric_id=metric_id, spec=spec
        )
        if rows:
            await writer.record_results(rows)
        failed = sum(1 for r in rows if r.status is ResultStatus.FAILED)
        if not rows or failed == len(rows):
            status = RunStatus.FAILED
        elif failed:
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.SUCCEEDED
        await writer.finish_run(run_pk, status=status)
        if status in (RunStatus.SUCCEEDED, RunStatus.PARTIAL):
            try:
                await writer.refresh_bucket(run_pk, period_seconds=DAILY_PERIOD_SECONDS)
            except Exception:
                logger.warning("refresh_bucket_failed", provider=spec.provider, exc_info=True)
        return status
    except Exception as exc:
        if run_pk is not None:
            try:
                await writer.finish_run(run_pk, status=RunStatus.FAILED, error=str(exc))
            except Exception:
                logger.warning(
                    "finish_run_failed", provider=spec.provider, run_id=run_pk, exc_info=True
                )
        logger.warning("provider_fetch_failed", provider=spec.provider, error=str(exc))
        return RunStatus.FAILED


async def fetch_and_write_v2v(settings: Settings | None = None) -> dict[str, RunStatus]:
    """Fetch each provider's latest run into its own run row; return per-provider status.

    Mirrors the STT/TTS orchestrator: per-provider ``finish_run``, then a best-effort
    bucket/matview refresh that never fails the job. No cross-run dedup -- a same-day
    retry re-aggregates the slot, matching the orchestrator's behaviour.
    """
    settings = settings or get_settings()

    metric_id = settings.coval_s2s_latency_metric_id
    if not metric_id:
        raise RuntimeError("coval_s2s_latency_metric_id is not set")

    epoch = int(datetime.now(tz=UTC).timestamp())
    scheduled_at = datetime.fromtimestamp(epoch - epoch % DAILY_PERIOD_SECONDS, tz=UTC)

    async with _client(settings) as client, lifespan_pool(settings) as pool:
        writer = RunWriter(pool)
        statuses: dict[str, RunStatus] = {}
        for spec in AGENTS:
            agent_id = getattr(settings, spec.agent_id_attr)
            if not agent_id:
                logger.warning("agent_id_unset", provider=spec.provider, attr=spec.agent_id_attr)
                continue
            statuses[spec.provider] = await _fetch_one_provider(
                client,
                writer,
                spec=spec,
                agent_id=agent_id,
                metric_id=metric_id,
                runner_sha=settings.runner_sha,
                scheduled_at=scheduled_at,
            )

        if any(s in (RunStatus.SUCCEEDED, RunStatus.PARTIAL) for s in statuses.values()):
            try:
                await writer.refresh_stats_matviews()
            except Exception:
                logger.warning("refresh_stats_matviews_failed", exc_info=True)
        logger.info("s2s_fetch_done", statuses={p: str(s) for p, s in statuses.items()})
        return statuses


@click.command(name="fetch-s2s")
def fetch_s2s() -> None:
    """Fetch S2S latency from Coval and write per-clip rows (daily Cloud Run Job)."""
    from coval_bench.logging import configure_logging

    settings = get_settings()
    configure_logging(level=settings.log_level)
    statuses = asyncio.run(fetch_and_write_v2v(settings))
    if not statuses or all(s is RunStatus.FAILED for s in statuses.values()):
        raise click.ClickException("s2s fetch failed for all providers")


if __name__ == "__main__":
    fetch_s2s()
