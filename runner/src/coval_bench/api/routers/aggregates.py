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

``/results/aggregates/by-dataset`` serves the per-dataset views (the WER
radar): every dataset's ``model_stats`` in one response, so a window toggle
costs one request instead of one per dataset. Series are not batched.
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
    has_enough_samples,
)
from coval_bench.api.deps import (
    capture_api_event,
    get_cache,
    get_cache_locks,
    get_pool,
    get_posthog,
)
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    AggregatesByDatasetResponse,
    AggregatesResponse,
    DatasetAggregates,
    ModelStatEntry,
    SeriesPoint,
    TimelinePoint,
    TimelineResponse,
)
from coval_bench.config import DATASET_ALL
from coval_bench.registries import is_metric_excluded

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["results"])

_STATS_SQL_TEMPLATE = (
    "SELECT provider, model, metric_type,"
    " avg_value, stddev_value, p25, p50, p75, p90, p95, p99,"
    " min_value, max_value, sample_count,"
    " wer_insertions_pct, wer_deletions_pct, wer_substitutions_pct"
    " FROM {view}"
    " WHERE benchmark = %(benchmark)s"
    " AND dataset_id = %(dataset)s"
    " ORDER BY provider, model, metric_type"
)

_SERIES_SQL = (
    "SELECT provider, model, metric_type, bucket_at AS scheduled_at,"
    " min_value, p25, p50, p75, max_value, value_sum, sample_count"
    " FROM benchmarks_v2.results_by_bucket"
    " WHERE benchmark = %(benchmark)s"
    " AND dataset_id = %(dataset)s"
    " AND bucket_at >= NOW() - %(interval)s::interval"
    " ORDER BY bucket_at, provider, model, metric_type"
)

_TIMELINE_SQL = (
    "SELECT provider, model, metric_type, bucket_at AS scheduled_at,"
    " CASE WHEN metric_type = 'WER'"
    " THEN value_sum / NULLIF(sample_count, 0) ELSE p50 END AS value"
    " FROM benchmarks_v2.results_by_bucket"
    " WHERE benchmark = %(benchmark)s"
    " AND dataset_id = %(dataset)s"
    " AND bucket_at >= NOW() - %(interval)s::interval"
    " ORDER BY bucket_at, provider, model, metric_type"
)

# Select a bounded, representative 30-day timeline in PostgreSQL.  Each exact
# provider/model/metric group is split into 119 ordinal bins; retaining the min
# and max plotted value in every bin plus the endpoints caps the result at 240
# points per group while preserving endpoints and plotted extrema.  All tie
# breaks include bucket_at so identical requests have identical ordering.
_COMPACT_SERIES_SQL = (
    "WITH base AS ("
    " SELECT provider, model, metric_type, bucket_at AS scheduled_at,"
    " min_value, p25, p50, p75, max_value, value_sum, sample_count,"
    " CASE WHEN metric_type = 'WER'"
    " THEN value_sum / NULLIF(sample_count, 0) ELSE p50 END AS value"
    " FROM benchmarks_v2.results_by_bucket"
    " WHERE benchmark = %(benchmark)s AND dataset_id = %(dataset)s"
    " AND bucket_at >= NOW() - %(interval)s::interval"
    "), ranked AS ("
    " SELECT *, row_number() OVER grp AS ordinal,"
    " count(*) OVER (PARTITION BY provider, model, metric_type) AS group_count"
    " FROM base WINDOW grp AS (PARTITION BY provider, model, metric_type ORDER BY scheduled_at)"
    "), binned AS ("
    " SELECT *, floor((ordinal - 1) * 119.0 / group_count)::int AS bin"
    " FROM ranked"
    "), selected AS ("
    " SELECT *, row_number() OVER (PARTITION BY provider, model, metric_type, bin"
    " ORDER BY value ASC, scheduled_at ASC) AS min_rank,"
    " row_number() OVER (PARTITION BY provider, model, metric_type, bin"
    " ORDER BY value DESC, scheduled_at ASC) AS max_rank"
    " FROM binned"
    ") SELECT provider, model, metric_type, scheduled_at, min_value, p25, p50, p75,"
    " max_value, value_sum, sample_count, value FROM selected"
    " WHERE min_rank = 1 OR max_rank = 1 OR ordinal = 1 OR ordinal = group_count"
    " ORDER BY scheduled_at, provider, model, metric_type"
)

_DATASETS_SQL_TEMPLATE = (
    "SELECT DISTINCT dataset_id FROM {view}"
    " WHERE benchmark = %(benchmark)s AND dataset_id <> %(sentinel)s"
    " ORDER BY dataset_id"
)

_STATS_BY_DATASET_SQL_TEMPLATE = (
    "SELECT dataset_id, provider, model, metric_type,"
    " avg_value, stddev_value, p25, p50, p75, p90, p95, p99,"
    " min_value, max_value, sample_count,"
    " wer_insertions_pct, wer_deletions_pct, wer_substitutions_pct"
    " FROM {view}"
    " WHERE benchmark = %(benchmark)s"
    " AND dataset_id <> %(sentinel)s"
    " ORDER BY dataset_id, provider, model, metric_type"
)


def _visible(row: dict[str, Any], hidden: frozenset[tuple[str, str]]) -> bool:
    return (row["provider"], row["model"]) not in hidden and not is_metric_excluded(
        row["provider"], row["model"], row["metric_type"]
    )


def _flag_thin(stat: ModelStatEntry, benchmark: str) -> ModelStatEntry:
    """Mark a stat that rests on too few samples to present as a real number.

    The values are left intact — a collapsed provider's one measurement is a real
    measurement, and callers that want it still get it. Only the flag changes, so
    the frontend can show "n/a" instead of ranking a lucky sample.
    """
    if has_enough_samples(benchmark, stat.sample_count):
        return stat
    return stat.model_copy(update={"insufficient_samples": True})


@router.get("/results/aggregates", response_model=AggregatesResponse)
@limiter.limit("60/minute")
async def get_results_aggregates(
    request: Request,  # required by slowapi
    benchmark: BenchmarkLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    dataset: str | None = Query(
        default=None,
        description="Dataset id to aggregate over; omit for the pooled all-dataset blocks.",
    ),
    include_series: bool = Query(
        default=True, description="Whether to include the per-bucket chart series."
    ),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
    cache: TTLCache[Any, Any] = Depends(get_cache),
    cache_locks: defaultdict[Any, asyncio.Lock] = Depends(get_cache_locks),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> AggregatesResponse:
    """Return per-model stats and per-bucket series for one benchmark.

    Args:
        benchmark: One of STT, TTS.
        window: Time window — stats over results.created_at, series over
            bucket_at. Defaults to 24h.
        dataset: Dataset id the blocks are computed over. Omitted, the pooled
            rows (every dataset together) are served — the pre-dataset-dimension
            behavior.
    """
    dataset_key = dataset or DATASET_ALL

    async def fill() -> AggregatesResponse:
        stats_sql = _STATS_SQL_TEMPLATE.format(view=WINDOW_VIEWS[window])
        datasets_sql = _DATASETS_SQL_TEMPLATE.format(view=WINDOW_VIEWS[window])
        stats_params: dict[str, Any] = {"benchmark": benchmark, "dataset": dataset_key}
        async with pool.connection() as conn:
            conn.row_factory = psycopg.rows.dict_row
            stat_rows = await (await conn.execute(stats_sql, stats_params)).fetchall()
            if include_series:
                series_rows = await (
                    await conn.execute(
                        _COMPACT_SERIES_SQL if window == "30d" else _SERIES_SQL,
                        {
                            "benchmark": benchmark,
                            "dataset": dataset_key,
                            "interval": WINDOW_INTERVALS[window],
                        },
                    )
                ).fetchall()
            else:
                series_rows = []
            dataset_rows = await (
                await conn.execute(datasets_sql, {"benchmark": benchmark, "sentinel": DATASET_ALL})
            ).fetchall()

        return AggregatesResponse(
            benchmark=benchmark,
            window=window,
            dataset=dataset_key,
            datasets=[r["dataset_id"] for r in dataset_rows],
            model_stats=[
                _flag_thin(ModelStatEntry.model_validate(r), benchmark)
                for r in stat_rows
                if _visible(r, hidden)
            ],
            # Series points are deliberately unflagged: one bucket holds a single
            # run's samples, so every point sits under the floor by design.
            series=[SeriesPoint.model_validate(r) for r in series_rows if _visible(r, hidden)],
        )

    # The hidden set is part of the key: two callers who can see different models
    # must never share a cache entry, or one would be served the other's rows.
    cache_key = (
        "aggregates",
        benchmark,
        window,
        dataset_key,
        include_series,
        tuple(sorted(hidden)),
    )
    response, cache_status = await get_or_fill(cache, cache_locks, cache_key, fill)

    capture_api_event(
        posthog_client,
        "results_aggregates_queried",
        {
            "benchmark": benchmark,
            "window": window,
            "dataset": dataset_key,
            "include_series": include_series,
            "model_stat_count": len(response.model_stats),
            "series_point_count": len(response.series),
            "cache_hit": cache_status != "miss",
            "cache_status": cache_status,
            "$process_person_profile": False,
        },
    )
    return response


@router.get("/results/timeline", response_model=TimelineResponse)
@limiter.limit("60/minute")
async def get_results_timeline(
    request: Request,
    benchmark: BenchmarkLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    dataset: str | None = Query(
        default=None,
        description="Dataset id to aggregate over; omit for pooled all-dataset buckets.",
    ),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
    cache: TTLCache[Any, Any] = Depends(get_cache),
    cache_locks: defaultdict[Any, asyncio.Lock] = Depends(get_cache_locks),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> TimelineResponse:
    """Return chart points without the dashboard's aggregate-stat payload."""
    dataset_key = dataset or DATASET_ALL

    async def fill() -> TimelineResponse:
        sql = _COMPACT_SERIES_SQL if window == "30d" else _TIMELINE_SQL
        params = {
            "benchmark": benchmark,
            "dataset": dataset_key,
            "interval": WINDOW_INTERVALS[window],
        }
        async with pool.connection() as conn:
            conn.row_factory = psycopg.rows.dict_row
            rows = await (await conn.execute(sql, params)).fetchall()
        return TimelineResponse(
            benchmark=benchmark,
            window=window,
            dataset=dataset_key,
            points=[
                TimelinePoint.model_validate(
                    row
                    if "value" in row
                    else {
                        **row,
                        "value": row["value_sum"] / row["sample_count"]
                        if row["metric_type"] == "WER"
                        else row["p50"],
                    }
                )
                for row in rows
                if _visible(row, hidden)
            ],
        )

    cache_key = ("timeline", benchmark, window, dataset_key, tuple(sorted(hidden)))
    response, cache_status = await get_or_fill(cache, cache_locks, cache_key, fill)
    capture_api_event(
        posthog_client,
        "results_timeline_queried",
        {
            "benchmark": benchmark,
            "window": window,
            "dataset": dataset_key,
            "point_count": len(response.points),
            "max_points_per_group": 240 if window == "30d" else None,
            "cache_hit": cache_status != "miss",
            "cache_status": cache_status,
            "$process_person_profile": False,
        },
    )
    return response


@router.get("/results/aggregates/by-dataset", response_model=AggregatesByDatasetResponse)
@limiter.limit("60/minute")
async def get_results_aggregates_by_dataset(
    request: Request,  # required by slowapi
    benchmark: BenchmarkLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
    cache: TTLCache[Any, Any] = Depends(get_cache),
    cache_locks: defaultdict[Any, asyncio.Lock] = Depends(get_cache_locks),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> AggregatesByDatasetResponse:
    """Return per-model stats for every dataset of one benchmark and window.

    One block per dataset with data in the window, sorted by dataset id. The
    pooled all-dataset rows are not repeated here — the plain aggregates
    endpoint serves those.
    """

    async def fill() -> AggregatesByDatasetResponse:
        stats_sql = _STATS_BY_DATASET_SQL_TEMPLATE.format(view=WINDOW_VIEWS[window])
        params = {"benchmark": benchmark, "sentinel": DATASET_ALL}

        async with pool.connection() as conn:
            conn.row_factory = psycopg.rows.dict_row
            rows = await (await conn.execute(stats_sql, params)).fetchall()

        grouped: dict[str, list[ModelStatEntry]] = {}
        for row in rows:
            if _visible(row, hidden):
                grouped.setdefault(row["dataset_id"], []).append(
                    _flag_thin(ModelStatEntry.model_validate(row), benchmark)
                )

        return AggregatesByDatasetResponse(
            benchmark=benchmark,
            window=window,
            blocks=[
                DatasetAggregates(dataset=dataset, model_stats=stats)
                for dataset, stats in grouped.items()
            ],
        )

    # The hidden set is part of the key: two callers who can see different models
    # must never share a cache entry, or one would be served the other's rows.
    cache_key = ("aggregates_by_dataset", benchmark, window, tuple(sorted(hidden)))
    response, cache_status = await get_or_fill(cache, cache_locks, cache_key, fill)

    capture_api_event(
        posthog_client,
        "results_aggregates_by_dataset_queried",
        {
            "benchmark": benchmark,
            "window": window,
            "dataset_count": len(response.blocks),
            "model_stat_count": sum(len(b.model_stats) for b in response.blocks),
            "cache_hit": cache_status != "miss",
            "cache_status": cache_status,
            "$process_person_profile": False,
        },
    )
    return response
