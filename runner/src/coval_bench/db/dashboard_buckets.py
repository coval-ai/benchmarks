# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Closed 1h/4h dashboard rollups, each filled once; delete its fill row to refill."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

BUCKET_INTERVALS: tuple[int, ...] = (3600, 14400)
RETENTION = timedelta(days=30)
CLOSE_GRACE = timedelta(minutes=5)
_FILL_STATEMENT_TIMEOUT = "300s"


@dataclass(frozen=True)
class FillResult:
    filled: int
    remaining: int


def floor_bucket(value: datetime, interval_seconds: int) -> datetime:
    """Align *value* to the start of its UTC bucket."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    epoch = int(value.timestamp())
    return datetime.fromtimestamp(epoch - epoch % interval_seconds, tz=UTC)


BUCKET_LOCK_SQL = """
SELECT pg_advisory_xact_lock(%(interval)s::int,
                             (extract(epoch FROM %(bucket)s::timestamptz) / 3600)::int)
"""

MISSING_BUCKETS_SQL = """
WITH candidates AS (
  SELECT generate_series(%(first)s::timestamptz, %(last)s::timestamptz, %(step)s) AS bucket_at
)
SELECT c.bucket_at
FROM candidates c
WHERE c.bucket_at + %(step)s + %(grace)s <= %(as_of)s::timestamptz
  AND NOT EXISTS (
    SELECT 1 FROM benchmarks_v2.dashboard_bucket_fills f
    WHERE f.interval_seconds = %(interval)s AND f.bucket_at = c.bucket_at)
  AND NOT EXISTS (
    SELECT 1 FROM benchmarks_v2.runs r
    WHERE r.status = 'running'
      AND r.scheduled_at >= c.bucket_at
      AND r.scheduled_at < c.bucket_at + %(step)s)
ORDER BY c.bucket_at
"""

FILL_BUCKET_SQL = """
INSERT INTO benchmarks_v2.dashboard_bucket_aggregates
(provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant,
 value_key, interval_seconds, bucket_at,
 min_value, p25, p50, p75, p90, p95, max_value, value_sum, sample_count,
 error_word_sum, reference_word_sum, latest_source_at)
WITH evaluations AS (
  SELECT o.provider, o.model, o.benchmark, o.dataset_id,
         e.metric_id, e.metric_version, e.evaluation_variant,
         e.value, e.roundtrip, e.leading_silence,
         e.substitution_count + e.deletion_count + e.insertion_count AS error_words,
         e.reference_words, r.scheduled_at
  FROM benchmarks_v2.dashboard_metric_values e
  JOIN benchmarks_v2.benchmark_observations o ON o.id = e.observation_id
  JOIN benchmarks_v2.runs r ON r.id = o.run_id
  WHERE o.status = 'succeeded' AND r.status IN ('succeeded', 'partial')
    AND r.scheduled_at >= %(bucket)s AND r.scheduled_at < %(bucket)s + %(step)s
), measurements AS (
  SELECT e.*, part.value_key, part.measured
  FROM evaluations e
  CROSS JOIN LATERAL (VALUES
    ('primary', e.value), ('roundtrip', e.roundtrip), ('leading_silence', e.leading_silence)
  ) AS part(value_key, measured)
  WHERE part.measured IS NOT NULL
)
SELECT provider, model, benchmark, COALESCE(dataset_id, '__all__'),
       metric_id, metric_version, evaluation_variant, value_key, %(interval)s, %(bucket)s,
       MIN(measured)::float8,
       PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY measured)::float8,
       PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY measured)::float8,
       PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY measured)::float8,
       PERCENTILE_CONT(.9) WITHIN GROUP (ORDER BY measured)::float8,
       PERCENTILE_CONT(.95) WITHIN GROUP (ORDER BY measured)::float8,
       MAX(measured)::float8, SUM(measured)::float8, COUNT(*)::int,
       CASE WHEN value_key = 'primary' AND COUNT(error_words) = COUNT(*)
             AND COUNT(reference_words) = COUNT(*) THEN SUM(error_words)::float8 END,
       CASE WHEN value_key = 'primary' AND COUNT(error_words) = COUNT(*)
             AND COUNT(reference_words) = COUNT(*) THEN SUM(reference_words)::float8 END,
       MAX(scheduled_at)
FROM measurements
GROUP BY GROUPING SETS (
  (provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant,
   value_key),
  (provider, model, benchmark, metric_id, metric_version, evaluation_variant, value_key)
)
"""


async def missing_buckets(
    pool: AsyncConnectionPool[Any], *, interval_seconds: int, as_of: datetime
) -> list[datetime]:
    """Closed buckets inside the retention window that have never been filled."""
    params = {
        "interval": interval_seconds,
        "step": timedelta(seconds=interval_seconds),
        "first": floor_bucket(as_of - RETENTION, interval_seconds),
        "last": floor_bucket(as_of, interval_seconds),
        "grace": CLOSE_GRACE,
        "as_of": as_of,
    }
    async with pool.connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(MISSING_BUCKETS_SQL, params)
        return [row["bucket_at"] for row in await cur.fetchall()]


async def fill_bucket(
    pool: AsyncConnectionPool[Any], *, interval_seconds: int, bucket_at: datetime
) -> None:
    """Replace one bucket from raw observations and record it as filled."""
    bucket = floor_bucket(bucket_at, interval_seconds)
    params = {
        "interval": interval_seconds,
        "step": timedelta(seconds=interval_seconds),
        "bucket": bucket,
    }
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(f"SET LOCAL statement_timeout = '{_FILL_STATEMENT_TIMEOUT}'")
        await conn.execute(BUCKET_LOCK_SQL, params)
        await conn.execute(
            """DELETE FROM benchmarks_v2.dashboard_bucket_aggregates
               WHERE interval_seconds = %(interval)s AND bucket_at = %(bucket)s""",
            params,
        )
        await conn.execute(FILL_BUCKET_SQL, params)
        await conn.execute(
            """INSERT INTO benchmarks_v2.dashboard_bucket_fills (interval_seconds, bucket_at)
               VALUES (%(interval)s, %(bucket)s)
               ON CONFLICT (interval_seconds, bucket_at) DO UPDATE SET filled_at = now()""",
            params,
        )


async def fill_closed_buckets(
    pool: AsyncConnectionPool[Any], *, as_of: datetime, deadline: float | None = None
) -> FillResult:
    """Fill missing closed buckets oldest-first, committing one at a time."""
    filled = 0
    remaining = 0
    loop = asyncio.get_running_loop()
    for interval_seconds in BUCKET_INTERVALS:
        pending = await missing_buckets(pool, interval_seconds=interval_seconds, as_of=as_of)
        for index, bucket_at in enumerate(pending):
            if deadline is not None and loop.time() >= deadline:
                remaining += len(pending) - index
                break
            await fill_bucket(pool, interval_seconds=interval_seconds, bucket_at=bucket_at)
            filled += 1
    return FillResult(filled, remaining)
