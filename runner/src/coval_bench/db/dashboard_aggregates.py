# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hourly dashboard maintenance: fill closed rollup buckets, then publish summaries."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from coval_bench.db.dashboard_buckets import fill_closed_buckets
from coval_bench.db.dashboard_source import DashboardPool
from coval_bench.db.dashboard_summaries import RefreshResult, refresh_summary_snapshots

logger = structlog.get_logger(__name__)

_MAINTENANCE_TIMEOUT_SECONDS = 540
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
