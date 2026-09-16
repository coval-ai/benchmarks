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

from coval_bench.db.dashboard_hourly import pending_hourly_aggregates, refresh_hourly_aggregates
from coval_bench.db.dashboard_source import DashboardPool, rebuild_source_bucket
from coval_bench.db.dashboard_summaries import RefreshResult, refresh_summary_snapshots

logger = structlog.get_logger(__name__)

_MAINTENANCE_TIMEOUT_SECONDS = 540


@dataclass(frozen=True)
class MaintenanceResult:
    source_buckets: int
    hourly_buckets: int
    summary: RefreshResult


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
        failures.append(exc)
        logger.error("dashboard_source_reconciliation_failed", exc_info=True)
    try:
        async with asyncio.timeout(120):
            pending_hours = await pending_hourly_aggregates(
                pool, since=at - timedelta(days=30), until=at
            )
            # Commit one hour at a time so a timeout retains completed repairs.
            for hour in pending_hours:
                hours += await refresh_hourly_aggregates(pool, hours=[hour])
    except Exception as exc:
        failures.append(exc)
        logger.error("dashboard_hourly_reconciliation_failed", exc_info=True)
    summary = RefreshResult("skipped_lock")
    try:
        async with asyncio.timeout_at(deadline):
            summary = await refresh_summary_snapshots(pool, as_of=at)
    except Exception as exc:
        failures.append(exc)
        logger.error("dashboard_summary_reconciliation_failed", exc_info=True)
    if failures:
        raise RuntimeError(
            "dashboard maintenance incomplete; pending work is retained"
        ) from failures[0]
    return MaintenanceResult(sources, hours, summary)


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
