# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hourly dashboard maintenance: drain the rollup queue, then refresh the window views."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from coval_bench.db.dashboard_rollups import DashboardPool, drain_rollup_queue
from coval_bench.db.dashboard_windows import RefreshResult, refresh_window_views

logger = structlog.get_logger(__name__)

_MAINTENANCE_TIMEOUT_SECONDS = 540
_SUMMARY_RESERVE_SECONDS = 120
_FILL_SLACK_SECONDS = 60


@dataclass(frozen=True)
class MaintenanceResult:
    filled: int
    remaining: int
    summary: RefreshResult
    elapsed: float = 0.0


async def refresh_dashboard_aggregates(
    pool: DashboardPool, *, as_of: datetime | None = None
) -> MaintenanceResult:
    """Rebuild every queued run slot the budget allows, then refresh the window views."""
    at = as_of or datetime.now(UTC)
    if at.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + _MAINTENANCE_TIMEOUT_SECONDS
    failures: list[Exception] = []
    filled = 0
    remaining = 0
    fill_deadline = deadline - _SUMMARY_RESERVE_SECONDS
    try:
        async with asyncio.timeout_at(fill_deadline + _FILL_SLACK_SECONDS):
            result = await drain_rollup_queue(pool, as_of=at, deadline=fill_deadline)
        filled, remaining = result.rebuilt, result.remaining
    except Exception as exc:
        failures.append(exc)
        logger.error("dashboard_rollup_fill_failed", exc_info=True)
    summary = RefreshResult("skipped_lock")
    try:
        async with asyncio.timeout_at(deadline):
            summary = await refresh_window_views(pool, as_of=at)
    except Exception as exc:
        failures.append(exc)
        logger.error("dashboard_window_refresh_failed", exc_info=True)
    elapsed = loop.time() - started
    logger.info(
        "dashboard_rollup_fill_completed",
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
