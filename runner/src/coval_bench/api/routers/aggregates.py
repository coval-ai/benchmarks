# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/results/aggregates — server-side dashboard aggregation.

Serves the dashboard's chart data as pre-computed aggregates. Two blocks:

* ``model_stats`` — per (provider, model, metric_type): avg, sample stddev
  (n-1 denominator, coalesced to 0 for n=1), p25/p50/p75 (percentile_cont),
  min, max, count.
* ``series`` — per (provider, model, metric_type, time bucket): avg and count.
  At 24h the bucket is the run's scheduled_at, falling back to created_at
  floored to the scheduler period for legacy rows, identical to the COALESCE
  in GET /v1/results. The wider windows use fixed buckets (7d: 2h, 30d: 12h)
  so the series payload stays near its 24h size.

Both blocks gate on result rows with status='success' and a non-null
metric_value, from parent runs in (succeeded, partial).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import psycopg.rows
import structlog
from cachetools import TTLCache
from fastapi import APIRouter, Depends, Query
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.cache import get_or_fill
from coval_bench.api.common import (
    SCHEDULED_AT_BUCKET_SQL,
    WINDOWS,
    BenchmarkLiteral,
    WindowLiteral,
    floor_epoch_sql,
)
from coval_bench.api.deps import (
    capture_api_event,
    get_cache,
    get_cache_locks,
    get_pool,
    get_posthog,
    get_settings,
)
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import AggregatesResponse, ModelStatEntry, SeriesPoint
from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["results"])

# Shared FROM/WHERE for both blocks.
_FROM_WHERE = (
    " FROM benchmarks_v2.results r"
    " JOIN benchmarks_v2.runs rn ON rn.id = r.run_id"
    " WHERE r.status = 'success'"
    " AND rn.status IN ('succeeded', 'partial')"
    " AND r.benchmark = %(benchmark)s"
    " AND r.metric_value IS NOT NULL"
    " AND r.created_at >= NOW() - %(interval)s::interval"
)

_STATS_SQL = (
    "SELECT r.provider, r.model, r.metric_type,"
    " AVG(r.metric_value)::float8 AS avg_value,"
    " COALESCE(STDDEV_SAMP(r.metric_value), 0)::float8 AS stddev_value,"
    " PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY r.metric_value)::float8 AS p25,"
    " PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.metric_value)::float8 AS p50,"
    " PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.metric_value)::float8 AS p75,"
    " MIN(r.metric_value)::float8 AS min_value,"
    " MAX(r.metric_value)::float8 AS max_value,"
    " COUNT(*)::int AS sample_count" + _FROM_WHERE + " GROUP BY r.provider, r.model, r.metric_type"
    " ORDER BY r.provider, r.model, r.metric_type"
)

# Wider windows floor scheduled_at (created_at for legacy rows) to the bucket.
_COARSE_BUCKET_SQL = floor_epoch_sql("COALESCE(rn.scheduled_at, r.created_at)", "bucket_seconds")


def _build_series_sql(bucket_sql: str) -> str:
    return (
        "SELECT r.provider, r.model, r.metric_type,"
        f" {bucket_sql} AS scheduled_at,"
        " AVG(r.metric_value)::float8 AS avg_value,"
        " COUNT(*)::int AS sample_count"
        + _FROM_WHERE
        + f" GROUP BY r.provider, r.model, r.metric_type, {bucket_sql}"
        " ORDER BY 4, r.provider, r.model, r.metric_type"
    )


_SERIES_SQL = _build_series_sql(SCHEDULED_AT_BUCKET_SQL)
_COARSE_SERIES_SQL = _build_series_sql(_COARSE_BUCKET_SQL)


@router.get("/results/aggregates", response_model=AggregatesResponse)
@limiter.limit("60/minute")
async def get_results_aggregates(
    request: Request,  # required by slowapi
    benchmark: BenchmarkLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    settings: Settings = Depends(get_settings),
    posthog_client: Posthog | None = Depends(get_posthog),
    cache: TTLCache[Any, Any] = Depends(get_cache),
    cache_locks: defaultdict[Any, asyncio.Lock] = Depends(get_cache_locks),
) -> AggregatesResponse:
    """Return per-model stats and per-bucket series for one benchmark.

    Args:
        benchmark: One of STT, TTS.
        window: Time window over results.created_at. Defaults to 24h.
    """

    async def fill() -> AggregatesResponse:
        spec = WINDOWS[window]
        params: dict[str, Any] = {"benchmark": benchmark, "interval": spec.interval}
        if spec.series_bucket_seconds is None:
            series_sql = _SERIES_SQL
            series_params: dict[str, Any] = {
                **params,
                "schedule_period": settings.schedule_period_seconds,
            }
        else:
            series_sql = _COARSE_SERIES_SQL
            series_params = {**params, "bucket_seconds": spec.series_bucket_seconds}

        async with pool.connection() as conn:
            conn.row_factory = psycopg.rows.dict_row
            stat_rows = await (await conn.execute(_STATS_SQL, params)).fetchall()
            series_rows = await (await conn.execute(series_sql, series_params)).fetchall()

        return AggregatesResponse(
            benchmark=benchmark,
            window=window,
            model_stats=[ModelStatEntry.model_validate(r) for r in stat_rows],
            series=[SeriesPoint.model_validate(r) for r in series_rows],
        )

    cache_key = ("aggregates", benchmark, window)
    response, cache_status = await get_or_fill(cache, cache_locks, cache_key, fill)

    capture_api_event(
        posthog_client,
        "results_aggregates_queried",
        {
            "benchmark": benchmark,
            "window": window,
            "model_stat_count": len(response.model_stats),
            "series_point_count": len(response.series),
            "cache_hit": cache_status != "miss",
            "cache_status": cache_status,
            "$process_person_profile": False,
        },
    )
    return response
