# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hourly dashboard maintenance: fill closed rollup buckets, then publish summaries."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg
import psycopg.rows
import structlog
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.dashboard_buckets import fill_buckets_covering, fill_closed_buckets
from coval_bench.db.dashboard_source import DashboardPool, rebuild_source_bucket
from coval_bench.db.dashboard_summaries import RefreshResult, refresh_summary_snapshots

logger = structlog.get_logger(__name__)

_MAINTENANCE_TIMEOUT_SECONDS = 540
# Summaries always get this much of the budget, even when the fill backlog is large.
_SUMMARY_RESERVE_SECONDS = 120


@dataclass(frozen=True)
class MaintenanceResult:
    filled: int
    remaining: int
    summary: RefreshResult
    elapsed: float = 0.0


async def reconcile_dashboard_aggregates(
    pool: DashboardPool, *, as_of: datetime | None = None
) -> MaintenanceResult:
    """Fill every missing closed bucket the budget allows, then refresh summaries."""
    at = as_of or datetime.now(UTC)
    if at.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + _MAINTENANCE_TIMEOUT_SECONDS
    failures: list[Exception] = []
    filled = 0
    remaining = 0
    try:
        async with asyncio.timeout_at(deadline - _SUMMARY_RESERVE_SECONDS):
            result = await fill_closed_buckets(
                pool, as_of=at, deadline=deadline - _SUMMARY_RESERVE_SECONDS
            )
        filled, remaining = result.filled, result.remaining
    except Exception as exc:
        failures.append(exc)
        logger.error("dashboard_bucket_fill_failed", exc_info=True)
    summary = RefreshResult("skipped_lock")
    try:
        async with asyncio.timeout_at(deadline):
            summary = await refresh_summary_snapshots(pool, as_of=at)
    except Exception as exc:
        failures.append(exc)
        logger.error("dashboard_summary_reconciliation_failed", exc_info=True)
    elapsed = loop.time() - started
    logger.info(
        "dashboard_bucket_fill_completed",
        filled=filled,
        remaining=remaining,
        elapsed=elapsed,
        summary=summary.status,
    )
    if failures:
        raise RuntimeError("dashboard maintenance incomplete; missing buckets are retained") from (
            failures[0]
        )
    return MaintenanceResult(filled, remaining, summary, elapsed)


async def repair_dashboard_aggregates(
    pool: DashboardPool, *, buckets: Sequence[datetime], as_of: datetime | None = None
) -> MaintenanceResult:
    """Rebuild explicit run slots, refill the closed buckets containing them, republish."""
    unique = sorted(set(buckets))
    if not unique or any(bucket.tzinfo is None for bucket in unique):
        raise ValueError("repair requires timezone-aware source bucket timestamps")
    for bucket in unique:
        await rebuild_source_bucket(pool, bucket)
    filled = await fill_buckets_covering(pool, unique, as_of=as_of)
    summary = await refresh_summary_snapshots(pool, as_of=as_of)
    return MaintenanceResult(filled, 0, summary)


def refresh_backfilled_dashboard(
    connection: psycopg.Connection[Any], *, buckets: Sequence[datetime]
) -> None:
    """Refill closed buckets and summaries after a synchronous backfill has committed."""
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
            await fill_buckets_covering(pool, buckets)
            await refresh_summary_snapshots(pool)

    asyncio.run(publish())
