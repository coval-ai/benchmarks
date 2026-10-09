# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/results/aggregates — server-side dashboard aggregation.

Serves the dashboard's chart data as pre-computed aggregates. Two blocks:

* ``model_stats`` — per (provider, model, metric_type): avg, sample stddev
  (n-1 denominator, coalesced to 0 for n=1), p25/p50/p75/p90/p95/p99
  (percentile_cont), min, max, count. Read from the per-window materialized
  views (``results_24h``/``results_7d``/``results_30d``), refreshed by the
  runner at the end of each benchmark run — read-only here. Normalized reads
  read atomically published snapshots built from the transactional value projection,
  with observation/run eligibility and window boundaries fixed at publication.
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
import datetime as dt
from collections import defaultdict
from typing import Any

import structlog
from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.cache import get_or_fill
from coval_bench.api.common import (
    LEGACY_WINDOW_VIEWS,
    WINDOW_INTERVALS,
    BenchmarkLiteral,
    StatisticLiteral,
    WindowLiteral,
    has_enough_samples,
    reads_normalized,
)
from coval_bench.api.dashboard_windows import dashboard_read, require_window_state
from coval_bench.api.deps import (
    capture_api_event,
    get_cache,
    get_cache_locks,
    get_pool,
    get_posthog,
    get_settings,
)
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    AggregatesByDatasetResponse,
    AggregatesResponse,
    DatasetAggregates,
    DatasetPersona,
    ModelStatEntry,
    SeriesPoint,
    TimelinePoint,
    TimelineResponse,
)
from coval_bench.config import DATASET_ALL, Settings
from coval_bench.db.dashboard_rollups import grain_for_seconds
from coval_bench.db.dashboard_windows import WINDOW_VIEWS
from coval_bench.registries import (
    METRIC_SPECS,
    Metric,
    is_metric_excluded,
)

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

_SAVED_STATS_SQL_TEMPLATE = (
    _STATS_SQL_TEMPLATE.replace(
        "SELECT provider, model, metric_type,",
        "SELECT provider, model, m.code AS metric_type,",
    )
    .replace(
        " FROM {view}",
        " FROM {view} v JOIN benchmarks_v2.metrics m ON m.id = v.metric_id",
    )
    .replace(
        " avg_value,",
        " mean_value, pooled_value, pooled_insertions_pct, pooled_deletions_pct,"
        " pooled_substitutions_pct, avg_value,",
    )
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
_COMPACT_SERIES_TAIL = (
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
    " max_value, value_sum, sample_count, value, error_sum, reference_word_sum, pooled_value"
    " FROM selected"
    " WHERE min_rank = 1 OR max_rank = 1 OR ordinal = 1 OR ordinal = group_count"
    " ORDER BY scheduled_at, provider, model, metric_type"
)  # noqa: S608

_COMPACT_SERIES_SQL = (
    "WITH base AS ("  # noqa: S608
    " SELECT provider, model, metric_type, bucket_at AS scheduled_at,"
    " min_value, p25, p50, p75, max_value, value_sum, sample_count,"
    " CASE WHEN metric_type = 'WER'"
    " THEN value_sum / NULLIF(sample_count, 0) ELSE p50 END AS value,"
    " NULL::float8 AS error_sum, NULL::float8 AS reference_word_sum, NULL::float8 AS pooled_value"
    " FROM benchmarks_v2.results_by_bucket"
    " WHERE benchmark = %(benchmark)s AND dataset_id = %(dataset)s"
    " AND bucket_at >= NOW() - %(interval)s::interval" + _COMPACT_SERIES_TAIL
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
_SAVED_STATS_BY_DATASET_SQL_TEMPLATE = _STATS_BY_DATASET_SQL_TEMPLATE.replace(
    "SELECT dataset_id, provider, model, metric_type,",
    "SELECT v.dataset_id, v.provider, v.model, m.code AS metric_type,",
).replace(
    " FROM {view}",
    " FROM {view} v JOIN benchmarks_v2.metrics m ON m.id = v.metric_id",
)

_NORMALIZED_DATASETS_SQL = """
SELECT DISTINCT dataset_id
FROM {view}
WHERE benchmark = %(benchmark)s
  AND dataset_id <> %(sentinel)s
  AND primary_sample_count > 0
ORDER BY dataset_id
"""

_RUN_SLOT_SQL = """
SELECT b.provider, b.model, m.code AS metric_type, b.bucket_at AS scheduled_at,
       b.min_value, b.p25, b.p50, b.p75, b.max_value, b.value_sum, b.sample_count,
       b.wer_error_words AS error_sum, b.wer_reference_words AS reference_word_sum,
       100 * b.wer_error_words / NULLIF(b.wer_reference_words, 0) AS pooled_value
FROM benchmarks_v2.dashboard_rollups b
JOIN benchmarks_v2.metrics m ON m.id = b.metric_id
WHERE b.grain = 'run' AND b.value_key = 'primary'
  AND b.metric_version = 'v1' AND b.evaluation_variant = 'default'
  AND b.benchmark = %(benchmark)s AND b.dataset_id = %(dataset)s
  AND b.bucket_at >= NOW() - %(interval)s::interval
"""

_BUCKET_VALUE = (
    " CASE WHEN metric_type = 'WER'"
    " THEN COALESCE(pooled_value, value_sum / NULLIF(sample_count, 0)) ELSE p50 END AS value"
)

_NORMALIZED_SERIES_SQL = (
    "SELECT * FROM (" + _RUN_SLOT_SQL + ") b"  # noqa: S608
    " ORDER BY scheduled_at, provider, model, metric_type"
)

_NORMALIZED_TIMELINE_SQL = (
    "SELECT provider, model, metric_type, scheduled_at, pooled_value,"  # noqa: S608
    + _BUCKET_VALUE
    + " FROM ("
    + _RUN_SLOT_SQL
    + ") b"
    " ORDER BY scheduled_at, provider, model, metric_type"
)

_NORMALIZED_COMPACT_SERIES_SQL = (
    "WITH base AS (SELECT *,"  # noqa: S608
    + _BUCKET_VALUE
    + " FROM ("
    + _RUN_SLOT_SQL
    + ") b"
    + _COMPACT_SERIES_TAIL
)

# Rules are data bound through psycopg, including metric and component names.
# Source buckets retain sums/counts, so larger intervals never average averages.
_AVERAGE_HEAD_SQL = """
SELECT provider, model, metric_type, scheduled_at, sample_count, latest_source_at,
       COALESCE(pooled_value, value_sum / NULLIF(sample_count, 0)) AS value,
       pooled_value,
       CASE WHEN metric_type <> 'WER' THEN 'mean'
            WHEN pooled_value IS NOT NULL THEN 'ratio'
            ELSE 'mean_fallback' END AS aggregation_method
"""

# Only closed buckets exist, so the newest interval always lags by one bucket.
_ROLLUP_AVERAGE_SQL = f"""
{_AVERAGE_HEAD_SQL}
FROM (
 SELECT b.provider, b.model, m.code AS metric_type, b.bucket_at AS scheduled_at,
        b.value_sum, b.sample_count, b.latest_run_at AS latest_source_at,
        100 * b.wer_error_words / NULLIF(b.wer_reference_words, 0) AS pooled_value
 FROM benchmarks_v2.dashboard_rollups b
 JOIN benchmarks_v2.metrics m ON m.id = b.metric_id
 WHERE b.value_key = 'primary'
   AND b.metric_version = 'v1' AND b.evaluation_variant = 'default'
   AND b.benchmark = %(benchmark)s AND b.dataset_id = %(dataset)s
   AND b.grain = %(grain)s
   AND b.bucket_at >= %(since)s AND b.bucket_at + %(step)s <= %(until)s
) buckets ORDER BY scheduled_at, provider, model, metric_type
"""  # noqa: S608

_LEGACY_AVERAGE_SQL = f"""
{_AVERAGE_HEAD_SQL}
FROM (
 SELECT provider, model, metric_type,
        to_timestamp(floor(extract(epoch FROM bucket_at) / %(bucket_seconds)s)
                     * %(bucket_seconds)s) AS scheduled_at,
        SUM(value_sum)::float8 AS value_sum, SUM(sample_count) AS sample_count,
        MAX(bucket_at) AS latest_source_at, NULL::float8 AS pooled_value
 FROM benchmarks_v2.results_by_bucket
 WHERE benchmark = %(benchmark)s AND dataset_id = %(dataset)s
   AND bucket_at >= %(since)s AND bucket_at < %(until)s
 GROUP BY provider, model, metric_type, scheduled_at
) buckets ORDER BY scheduled_at, provider, model, metric_type
"""  # noqa: S608

_PERCENTILE_METRICS = frozenset(
    {"WER", "TTFT", "TTFS", "AudioToFinal", "TTFA", "TTFARoundtrip", "TTFALeadingSilence", "V2V"}
)
# TTFA's components are value keys on the TTFA row, not metrics of their own.
_PERCENTILE_VALUE_KEYS: dict[str, tuple[str, str]] = {
    "TTFARoundtrip": ("TTFA", "roundtrip"),
    "TTFALeadingSilence": ("TTFA", "leading_silence"),
}
_PERCENTILE_COLUMNS = {"p50": "b.p50", "p90": "b.p90", "p95": "b.p95"}

_PERCENTILE_SQL = """
SELECT b.provider, b.model, %(metric_type)s AS metric_type, b.bucket_at AS scheduled_at,
       {column} AS value, b.sample_count, b.latest_run_at AS latest_source_at,
       'percentile' AS aggregation_method
FROM benchmarks_v2.dashboard_rollups b
JOIN benchmarks_v2.metrics m ON m.id = b.metric_id
WHERE b.grain = %(grain)s
  AND b.metric_version = 'v1' AND b.evaluation_variant = 'default'
  AND b.benchmark = %(benchmark)s AND b.dataset_id = %(dataset)s
  AND m.code = %(base_metric)s AND b.value_key = %(value_key)s
  AND b.bucket_at >= %(since)s AND b.bucket_at + %(step)s <= %(until)s
ORDER BY scheduled_at, provider, model
"""


def _timeline_bucket_seconds(duration_seconds: float) -> int:
    """Hourly buckets up to roughly 200 points, four-hour buckets beyond that."""
    return 3600 if duration_seconds / 200.0 <= 3600 else 14400


def _percentile_sql(statistic: str) -> str:
    return _PERCENTILE_SQL.format(column=_PERCENTILE_COLUMNS[statistic])


def _validate_percentile_request(
    benchmark: str, statistic: StatisticLiteral, metric_type: str | None
) -> None:
    if statistic == "default":
        return
    if metric_type is None:
        raise HTTPException(
            status_code=422, detail="metric_type is required for percentile timelines"
        )
    if metric_type not in _PERCENTILE_METRICS or metric_type not in METRIC_SPECS:
        raise HTTPException(status_code=422, detail="unsupported percentile metric")
    if benchmark not in {item.value for item in METRIC_SPECS[Metric(metric_type)].benchmarks}:
        raise HTTPException(status_code=422, detail="metric is not supported for benchmark")


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
    settings: Settings = Depends(get_settings),
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
        normalized = reads_normalized(settings.normalized_dashboard_reads_enabled, benchmark)
        stats_sql = (
            _SAVED_STATS_SQL_TEMPLATE.format(view=WINDOW_VIEWS[window])
            if normalized
            else _STATS_SQL_TEMPLATE.format(view=LEGACY_WINDOW_VIEWS[window])
        )
        datasets_sql = (
            _NORMALIZED_DATASETS_SQL.format(view=WINDOW_VIEWS[window])
            if normalized
            else _DATASETS_SQL_TEMPLATE.format(view=LEGACY_WINDOW_VIEWS[window])
        )
        stats_params: dict[str, Any] = {
            "benchmark": benchmark,
            "dataset": dataset_key,
            "interval": WINDOW_INTERVALS[window],
        }
        async with dashboard_read(pool, saved=normalized) as conn:
            snapshot = await require_window_state(conn) if normalized else None
            stat_rows = await (await conn.execute(stats_sql, stats_params)).fetchall()
            if include_series:
                series_rows = await (
                    await conn.execute(
                        (
                            _NORMALIZED_COMPACT_SERIES_SQL
                            if window == "30d"
                            else _NORMALIZED_SERIES_SQL
                        )
                        if normalized
                        else (_COMPACT_SERIES_SQL if window == "30d" else _SERIES_SQL),
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
                await conn.execute(
                    datasets_sql,
                    {
                        "benchmark": benchmark,
                        "sentinel": DATASET_ALL,
                        "interval": WINDOW_INTERVALS[window],
                    },
                )
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
            snapshot=snapshot.as_dict() if snapshot else None,
        )

    # The hidden set is part of the key: two callers who can see different models
    # must never share a cache entry, or one would be served the other's rows.
    cache_key = (
        "aggregates",
        benchmark,
        window,
        dataset_key,
        include_series,
        settings.normalized_dashboard_reads_enabled,
        tuple(sorted(hidden)),
    )
    if reads_normalized(settings.normalized_dashboard_reads_enabled, benchmark):
        response, cache_status = await fill(), "bypass"
    else:
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
    window: WindowLiteral | None = Query(default=None),
    statistic: StatisticLiteral = Query(default="default"),
    metric_type: str | None = Query(default=None),
    dataset: str | None = Query(
        default=None,
        description="Dataset id to aggregate over; omit for pooled all-dataset buckets.",
    ),
    since: dt.datetime | None = Query(default=None),
    until: dt.datetime | None = Query(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
    cache: TTLCache[Any, Any] = Depends(get_cache),
    cache_locks: defaultdict[Any, asyncio.Lock] = Depends(get_cache_locks),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
    settings: Settings = Depends(get_settings),
) -> TimelineResponse:
    """Return run points or intervals using each metric's aggregation rule."""
    _validate_percentile_request(benchmark, statistic, metric_type)
    dataset_key = dataset or DATASET_ALL
    bucket_seconds: int | None
    if (since is None) != (until is None):
        raise HTTPException(status_code=422, detail="since and until must be provided together")
    if since is not None and until is not None:
        if window is not None:
            raise HTTPException(
                status_code=422, detail="window cannot be combined with since/until"
            )
        if since.tzinfo is None or until.tzinfo is None:
            raise HTTPException(status_code=422, detail="since and until must be timezone-aware")
        since = since.astimezone(dt.UTC)
        until = until.astimezone(dt.UTC)
        if until <= since:
            raise HTTPException(status_code=422, detail="until must be after since")
        duration = (until - since).total_seconds()
        if duration > 30 * 86400:
            raise HTTPException(status_code=422, detail="timeline range cannot exceed 30 days")
        response_window: WindowLiteral | None = None
        bucket_seconds = _timeline_bucket_seconds(duration)
        aggregation = "average"
    else:
        response_window = window or "24h"
        duration = {"24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}[response_window]
        now = dt.datetime.now(dt.UTC)
        since = now - dt.timedelta(seconds=duration)
        until = now
        bucket_seconds = _timeline_bucket_seconds(duration) if response_window != "24h" else None
        aggregation = "run" if response_window == "24h" else "average"

    async def fill() -> TimelineResponse:
        normalized = reads_normalized(settings.normalized_dashboard_reads_enabled, benchmark)
        rule_params: dict[str, Any] = {}
        if statistic != "default":
            sql = _percentile_sql(statistic)
            base_metric, value_key = _PERCENTILE_VALUE_KEYS.get(
                metric_type or "", (metric_type, "primary")
            )
            rule_params = {"base_metric": base_metric, "value_key": value_key}
        elif aggregation == "run":
            sql = _NORMALIZED_TIMELINE_SQL if normalized else _TIMELINE_SQL
        else:
            sql = _ROLLUP_AVERAGE_SQL if normalized else _LEGACY_AVERAGE_SQL
        params = {
            "benchmark": benchmark,
            "dataset": dataset_key,
            # Only the unchanged run query uses this interval; averages use
            # the explicit source bounds below.
            "interval": (
                f"{int(duration)} seconds"
                if response_window is None
                else WINDOW_INTERVALS[response_window]
            ),
            "since": since,
            "until": until,
            "bucket_seconds": bucket_seconds,
            "grain": grain_for_seconds(bucket_seconds),
            "step": dt.timedelta(seconds=bucket_seconds or 0),
            "metric_type": metric_type,
        }
        params.update(rule_params)
        # Percentiles only exist in normalized storage, whatever the read flag says.
        async with dashboard_read(pool, saved=normalized or statistic != "default") as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        visible_rows = [
            row
            for row in rows
            if _visible(row, hidden) and (metric_type is None or row["metric_type"] == metric_type)
        ]
        return TimelineResponse(
            benchmark=benchmark,
            statistic=statistic,
            metric_type=metric_type,
            precision="exact" if statistic != "default" else None,
            percentile_method="continuous" if statistic != "default" else None,
            weighting="observation" if statistic != "default" else None,
            window=response_window,
            dataset=dataset_key,
            points=[
                TimelinePoint.model_validate(
                    {
                        **row,
                        "insufficient_samples": statistic != "default"
                        and not has_enough_samples(benchmark, row.get("sample_count") or 0),
                    }
                )
                for row in visible_rows
            ],
            aggregation=aggregation,
            bucket_seconds=bucket_seconds,
            range_start=since,
            range_end=until,
            latest_source_at=max(
                (row.get("latest_source_at", row.get("scheduled_at")) for row in visible_rows),
                default=None,
            ),
        )

    cache_key = (
        "timeline",
        benchmark,
        response_window,
        dataset_key,
        aggregation,
        statistic,
        metric_type,
        bucket_seconds,
        # Preset keys intentionally omit request time; custom bounds are isolated.
        None if response_window is not None else (since, until),
        settings.normalized_dashboard_reads_enabled,
        tuple(sorted(hidden)),
    )
    response, cache_status = await get_or_fill(cache, cache_locks, cache_key, fill)
    capture_api_event(
        posthog_client,
        "results_timeline_queried",
        {
            "benchmark": benchmark,
            "window": response_window,
            "dataset": dataset_key,
            "aggregation": aggregation,
            "bucket_seconds": bucket_seconds,
            "point_count": len(response.points),
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
    settings: Settings = Depends(get_settings),
) -> AggregatesByDatasetResponse:
    """Return per-model stats for every dataset of one benchmark and window.

    One block per dataset with data in the window, sorted by dataset id. The
    pooled all-dataset rows are not repeated here — the plain aggregates
    endpoint serves those.
    """

    async def fill() -> AggregatesByDatasetResponse:
        normalized = reads_normalized(settings.normalized_dashboard_reads_enabled, benchmark)
        stats_sql = (
            _SAVED_STATS_BY_DATASET_SQL_TEMPLATE.replace(
                " avg_value,",
                " mean_value, pooled_value, pooled_insertions_pct, pooled_deletions_pct,"
                " pooled_substitutions_pct, avg_value,",
            ).format(view=WINDOW_VIEWS[window])
            if normalized
            else _STATS_BY_DATASET_SQL_TEMPLATE.format(view=LEGACY_WINDOW_VIEWS[window])
        )
        params = {
            "benchmark": benchmark,
            "sentinel": DATASET_ALL,
            "interval": WINDOW_INTERVALS[window],
        }

        async with dashboard_read(pool, saved=normalized) as conn:
            snapshot = await require_window_state(conn) if normalized else None
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
                DatasetAggregates(
                    dataset=dataset,
                    model_stats=stats,
                    persona=DatasetPersona.for_dataset(dataset),
                )
                for dataset, stats in grouped.items()
            ],
            snapshot=snapshot.as_dict() if snapshot else None,
        )

    # The hidden set is part of the key: two callers who can see different models
    # must never share a cache entry, or one would be served the other's rows.
    cache_key = (
        "aggregates_by_dataset",
        benchmark,
        window,
        settings.normalized_dashboard_reads_enabled,
        tuple(sorted(hidden)),
    )
    if reads_normalized(settings.normalized_dashboard_reads_enabled, benchmark):
        response, cache_status = await fill(), "bypass"
    else:
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
