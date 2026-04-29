# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/leaderboard — aggregated benchmark leaderboard.

Metric/benchmark compatibility:
- WER  + STT
- TTFT + STT
- TTFA + TTS

``window=24h`` queries the materialized view ``benchmarks_v2.results_24h``
(refreshed by a Cloud Scheduler cron in Phase 3 — read-only here).

``window=7d`` / ``window=30d`` run a live aggregation query against the
``results`` table directly.
"""

from __future__ import annotations

from typing import Any, Literal

import psycopg.rows
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import LeaderboardEntry, LeaderboardResponse

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["leaderboard"])

MetricLiteral = Literal["WER", "TTFA", "TTFT"]
BenchmarkLiteral = Literal["STT", "TTS"]
WindowLiteral = Literal["24h", "7d", "30d"]

# Valid (metric, benchmark) combinations — lower is better for all.
_VALID_COMBOS: set[tuple[str, str]] = {
    ("WER", "STT"),
    ("TTFT", "STT"),
    ("TTFA", "TTS"),
}

# Interval strings for live aggregation
_WINDOW_INTERVALS: dict[str, str] = {
    "7d": "7 days",
    "30d": "30 days",
}

# SQL for live aggregation (7d / 30d)
_LIVE_SQL = """
    SELECT provider, model,
           AVG(metric_value)::float8 AS avg,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY metric_value)::float8 AS p50,
           PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY metric_value)::float8 AS p95,
           COUNT(*)::int AS n
    FROM benchmarks_v2.results
    WHERE status = 'success'
      AND metric_type = %(metric)s
      AND benchmark = %(benchmark)s
      AND created_at >= NOW() - %(interval)s::interval
    GROUP BY provider, model
    ORDER BY avg ASC
"""

# SQL for materialized view (24h).
# Note: the MV stores ``avg_value`` (not ``avg``) — aliased here for LeaderboardEntry.
_MV_SQL = """
    SELECT provider, model,
           avg_value AS avg,
           p50,
           p95,
           n::int AS n
    FROM benchmarks_v2.results_24h
    WHERE metric_type = %(metric)s
      AND benchmark = %(benchmark)s
    ORDER BY avg_value ASC
"""


@router.get("/leaderboard", response_model=LeaderboardResponse)
@limiter.limit("60/minute")
async def get_leaderboard(
    request: Request,  # required by slowapi
    metric: MetricLiteral = Query(...),
    benchmark: BenchmarkLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> LeaderboardResponse:
    """Return leaderboard entries sorted ascending by average metric value.

    Args:
        metric: One of WER, TTFA, TTFT.
        benchmark: One of STT, TTS.
        window: Time window — 24h uses the materialized view; 7d/30d run live.

    Returns:
        ``{"metric": ..., "window": ..., "entries": [LeaderboardEntry, ...]}``

    Raises:
        400: If the metric/benchmark combination is incompatible.
    """
    if (metric, benchmark) not in _VALID_COMBOS:
        raise HTTPException(
            400,
            f"metric={metric!r} is not compatible with benchmark={benchmark!r}. "
            f"Valid combinations: WER+STT, TTFT+STT, TTFA+TTS.",
        )

    params: dict[str, Any] = {"metric": metric, "benchmark": benchmark}

    if window == "24h":
        sql = _MV_SQL
    else:
        sql = _LIVE_SQL
        params["interval"] = _WINDOW_INTERVALS[window]

    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = await conn.execute(sql, params)
        entry_rows = await rows.fetchall()

    entries = [LeaderboardEntry.model_validate(r) for r in entry_rows]
    return LeaderboardResponse(metric=metric, window=window, entries=entries)
