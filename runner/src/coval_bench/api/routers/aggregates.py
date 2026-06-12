# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/results/aggregates — server-side dashboard aggregation.

Serves the dashboard's chart data as pre-computed aggregates. Two blocks:

* ``model_stats`` — per (provider, model, metric_type): avg, sample stddev
  (n-1 denominator, coalesced to 0 for n=1), p25/p50/p75/p90/p95/p99
  (percentile_cont), min, max, count. Read from the per-window materialized
  views (``results_24h``/``results_7d``/``results_30d``), refreshed by the
  runner at the end of each benchmark run — read-only here.
* ``series`` — per (provider, model, metric_type, bucket_at) distribution
  (min/p25/p50/p75/max/value_sum/count), read from the ``results_by_bucket``
  rollup table, filled by the orchestrator's end-of-run hook.

Both blocks are pre-aggregated from rows with status='success' and a non-null
metric_value, from parent runs in (succeeded, partial) — read-only here.
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
    WINDOW_INTERVALS,
    WINDOW_VIEWS,
    BenchmarkLiteral,
    WindowLiteral,
)
from coval_bench.api.deps import (
    capture_api_event,
    get_cache,
    get_cache_locks,
    get_pool,
    get_posthog,
)
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import AggregatesResponse, ModelStatEntry, SeriesPoint

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["results"])

_STATS_SQL_TEMPLATE = (
    "SELECT provider, model, metric_type,"
    " avg_value, stddev_value, p25, p50, p75, p90, p95, p99,"
    " min_value, max_value, sample_count"
    " FROM {view}"
    " WHERE benchmark = %(benchmark)s"
    " ORDER BY provider, model, metric_type"
)

_SERIES_SQL = (
    "SELECT provider, model, metric_type, bucket_at AS scheduled_at,"
    " min_value, p25, p50, p75, max_value, value_sum, sample_count"
    " FROM benchmarks_v2.results_by_bucket"
    " WHERE benchmark = %(benchmark)s"
    " AND bucket_at >= NOW() - %(interval)s::interval"
    " ORDER BY bucket_at, provider, model, metric_type"
)


@router.get("/results/aggregates", response_model=AggregatesResponse)
@limiter.limit("60/minute")
async def get_results_aggregates(
    request: Request,  # required by slowapi
    benchmark: BenchmarkLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
    cache: TTLCache[Any, Any] = Depends(get_cache),
    cache_locks: defaultdict[Any, asyncio.Lock] = Depends(get_cache_locks),
) -> AggregatesResponse:
    """Return per-model stats and per-bucket series for one benchmark.

    Args:
        benchmark: One of STT, TTS.
        window: Time window — stats over results.created_at, series over
            bucket_at. Defaults to 24h.
    """

    async def fill() -> AggregatesResponse:
        stats_sql = _STATS_SQL_TEMPLATE.format(view=WINDOW_VIEWS[window])
        series_params: dict[str, Any] = {
            "benchmark": benchmark,
            "interval": WINDOW_INTERVALS[window],
        }

        async with pool.connection() as conn:
            conn.row_factory = psycopg.rows.dict_row
            stat_rows = await (await conn.execute(stats_sql, {"benchmark": benchmark})).fetchall()
            series_rows = await (await conn.execute(_SERIES_SQL, series_params)).fetchall()

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
