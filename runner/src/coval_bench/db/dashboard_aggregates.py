# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile saved dashboard data independently of benchmark execution."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import psycopg.rows
import structlog
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.dashboard_contracts import aggregation_fingerprint
from coval_bench.db.dashboard_hourly import (
    _refresh_hourly_aggregate,
    pending_hourly_aggregates,
    pending_hourly_status,
    prepare_hourly_aggregate_catalog,
    refresh_hourly_aggregates,
)
from coval_bench.db.dashboard_source import DashboardPool, rebuild_source_bucket
from coval_bench.db.dashboard_summaries import RefreshResult, refresh_summary_snapshots

logger = structlog.get_logger(__name__)

_MAINTENANCE_TIMEOUT_SECONDS = 540
_HOURLY_PHASE_TIMEOUT_SECONDS = 120
_HOURLY_BATCH_LIMIT = 24
_HOURLY_MIN_REMAINING_SECONDS = 5
_HOURLY_STATUS_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class MaintenanceResult:
    source_buckets: int
    hourly_buckets: int
    summary: RefreshResult
    selected: int = 0
    committed: int = 0
    remaining: int = 0
    oldest: datetime | None = None
    elapsed: float = 0.0
    stop_reason: str | None = None
    current_hour: datetime | None = None
    failure_stage: str | None = None
    hourly_failure_stage: str | None = None

    @property
    def hourly_selected(self) -> int:
        return self.selected

    @property
    def hourly_committed(self) -> int:
        return self.committed

    @property
    def hourly_remaining(self) -> int:
        return self.remaining

    @property
    def oldest_pending_hour(self) -> datetime | None:
        return self.oldest


async def reconcile_dashboard_aggregates(
    pool: DashboardPool, *, as_of: datetime | None = None
) -> MaintenanceResult:
    """Drain source/hour repairs and age summaries out even without new ingestion."""
    at = as_of or datetime.now(UTC)
    if at.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    deadline = asyncio.get_running_loop().time() + _MAINTENANCE_TIMEOUT_SECONDS
    failures: list[Exception] = []
    sources = 0
    hours = 0
    selected = 0
    remaining = 0
    oldest: datetime | None = None
    hourly_elapsed = 0.0
    hourly_stop_reason: str | None = None
    hourly_current: datetime | None = None
    failure_stage: str | None = None
    hourly_failures: list[Exception] = []
    hourly_owned_deadline = False
    safety_stop = False
    try:
        async with asyncio.timeout(240):
            async with pool.connection() as conn:
                cursor = await conn.execute(
                    "SELECT bucket_at FROM benchmarks_v2.dashboard_source_refreshes "
                    "ORDER BY bucket_at LIMIT 1000"
                )
                pending_sources = [row["bucket_at"] for row in await cursor.fetchall()]
            for bucket in pending_sources:
                await rebuild_source_bucket(pool, bucket)
                sources += 1
    except Exception as exc:
        failure_stage = "source"
        failures.append(exc)
        logger.error("dashboard_source_reconciliation_failed", exc_info=True)
    hourly_started = asyncio.get_running_loop().time()
    hourly_deadline = hourly_started + _HOURLY_PHASE_TIMEOUT_SECONDS
    hourly_timeout = asyncio.timeout(_HOURLY_PHASE_TIMEOUT_SECONDS)
    try:
        async with hourly_timeout:
            pending_hours = await pending_hourly_aggregates(
                pool,
                since=at - timedelta(days=30),
                until=at,
                limit=_HOURLY_BATCH_LIMIT,
            )
            selected = len(pending_hours)
            if pending_hours:
                metric_ids = await prepare_hourly_aggregate_catalog(pool)
                fingerprint = aggregation_fingerprint()
                # Commit one hour at a time so a timeout retains completed repairs.
                for hour in pending_hours:
                    if hourly_deadline - asyncio.get_running_loop().time() < (
                        _HOURLY_MIN_REMAINING_SECONDS
                    ):
                        safety_stop = True
                        break
                    hourly_current = hour
                    await _refresh_hourly_aggregate(
                        pool,
                        hour,
                        metric_ids=metric_ids,
                        fingerprint=fingerprint,
                    )
                    hours += 1
                    hourly_current = None
            else:
                hourly_stop_reason = "empty"
    except TimeoutError as exc:
        # asyncio.timeout() only owns a TimeoutError when its context reports
        # expiration. A TimeoutError raised by a statement or a dependency must
        # remain a genuine maintenance failure.
        if hourly_timeout.expired():
            hourly_owned_deadline = True
        else:
            failure_stage = "hourly"
            hourly_failure_stage = "hourly"
            hourly_failures.append(exc)
    except Exception as exc:
        failure_stage = "hourly"
        hourly_failure_stage = "hourly"
        hourly_failures.append(exc)
    finally:
        hourly_elapsed = asyncio.get_running_loop().time() - hourly_started

    status_needed = selected > 0 or hourly_owned_deadline or bool(hourly_failures) or safety_stop
    if status_needed:
        try:
            status_started = asyncio.get_running_loop().time()
            status_deadline = min(
                deadline,
                status_started + _HOURLY_STATUS_TIMEOUT_SECONDS,
            )
            if status_deadline <= status_started:
                raise RuntimeError("dashboard hourly backlog status deadline expired")
            async with asyncio.timeout_at(status_deadline):
                backlog = await pending_hourly_status(pool, since=at - timedelta(days=30), until=at)
            remaining = backlog.count
            oldest = backlog.oldest
        except Exception as exc:
            failure_stage = "hourly_status"
            hourly_failure_stage = "hourly_status"
            hourly_failures.append(RuntimeError("dashboard hourly backlog status unavailable"))
            hourly_failures[-1].__cause__ = exc

    if (hourly_owned_deadline or safety_stop) and hours == 0 and remaining > 0:
        failure_stage = "hourly"
        hourly_failure_stage = "hourly"
        hourly_failures.append(
            RuntimeError("dashboard hourly maintenance made no committed progress")
        )

    if hourly_failures:
        hourly_stop_reason = "failure"
    elif hourly_owned_deadline and remaining > 0:
        hourly_stop_reason = "deadline"
    elif safety_stop and remaining > 0:
        hourly_stop_reason = "safety_threshold"
    elif selected == _HOURLY_BATCH_LIMIT and hours == selected and remaining > 0:
        hourly_stop_reason = "batch_limit"
    elif remaining == 0:
        hourly_stop_reason = "complete"
    else:
        hourly_stop_reason = "backlog"
    failures.extend(hourly_failures)
    summary = RefreshResult("skipped_lock")
    try:
        async with asyncio.timeout_at(deadline):
            summary = await refresh_summary_snapshots(pool, as_of=at)
    except Exception as exc:
        failure_stage = "summary"
        failures.append(exc)
        logger.error("dashboard_summary_reconciliation_failed", exc_info=True)
    event_fields = {
        "selected": selected,
        "committed": hours,
        "remaining": remaining,
        "oldest": oldest.isoformat() if oldest is not None else None,
        "elapsed": hourly_elapsed,
        "stop_reason": hourly_stop_reason,
        "current_hour": hourly_current.isoformat() if hourly_current is not None else None,
        "failure_stage": hourly_failure_stage if hourly_failures else failure_stage,
    }
    if hourly_failures:
        logger.error("dashboard_hourly_reconciliation_failed", **event_fields)
    elif remaining:
        logger.info("dashboard_hourly_reconciliation_deferred", **event_fields)
    else:
        logger.info("dashboard_hourly_reconciliation_completed", **event_fields)
    if failures:
        raise RuntimeError(
            "dashboard maintenance incomplete; pending work is retained"
        ) from failures[0]
    # Preserve the compact result shape for the no-op path used by existing
    # callers; productive hourly runs return the detailed status fields.
    if selected == 0 and hours == 0 and remaining == 0:
        return MaintenanceResult(sources, hours, summary)
    return MaintenanceResult(
        sources,
        hours,
        summary,
        selected,
        hours,
        remaining,
        oldest,
        hourly_elapsed,
        hourly_stop_reason,
        hourly_current,
        failure_stage,
    )


async def repair_dashboard_aggregates(
    pool: DashboardPool, *, buckets: Sequence[datetime], as_of: datetime | None = None
) -> MaintenanceResult:
    """Repair explicit old/new source buckets, including buckets that became empty."""
    unique = sorted(set(buckets))
    if not unique or any(bucket.tzinfo is None for bucket in unique):
        raise ValueError("repair requires timezone-aware source bucket timestamps")
    async with pool.connection() as conn, conn.transaction(), conn.cursor() as cursor:
        await cursor.executemany(
            "INSERT INTO benchmarks_v2.dashboard_source_refreshes(bucket_at,requested_at) "
            "VALUES (%s,clock_timestamp()) ON CONFLICT (bucket_at) "
            "DO UPDATE SET requested_at=EXCLUDED.requested_at",
            [(bucket,) for bucket in unique],
        )
    for bucket in unique:
        await rebuild_source_bucket(pool, bucket)
    hours = await refresh_hourly_aggregates(pool, hours=unique)
    summary = await refresh_summary_snapshots(pool, as_of=as_of)
    return MaintenanceResult(len(unique), hours, summary)


def refresh_backfilled_dashboard(
    connection: psycopg.Connection[Any], *, buckets: Sequence[datetime]
) -> None:
    """Publish hours and summaries after a synchronous backfill has committed."""
    if not buckets:
        return

    async def publish() -> None:
        pool: DashboardPool = AsyncConnectionPool(
            conninfo=connection.info.dsn,
            min_size=1,
            max_size=2,
            open=False,
            # ConnectionInfo.dsn deliberately omits passwords. Preserve the
            # authenticated connection's password without serializing/logging it.
            kwargs={"row_factory": psycopg.rows.dict_row, "password": connection.info.password},
        )
        async with pool:
            await refresh_hourly_aggregates(pool, hours=buckets)
            await refresh_summary_snapshots(pool)

    asyncio.run(publish())
