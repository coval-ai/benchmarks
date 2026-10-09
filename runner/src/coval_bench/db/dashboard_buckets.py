# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Closed 1h/4h dashboard rollups computed once from raw observations.

A bucket is filled exactly once, after it closes: its end has passed and no run
scheduled inside it is still running.  Nothing is recomputed automatically; to
rebuild a range, delete its ``dashboard_bucket_fills`` rows and the next job run
fills it again.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

BUCKET_INTERVALS: tuple[int, ...] = (3600, 14400)
RETENTION = timedelta(days=30)
# Runs are created at their schedule slot, so a short grace only covers the
# scheduler firing a little late at a bucket boundary.
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


def is_closed(bucket_at: datetime, interval_seconds: int, *, as_of: datetime) -> bool:
    return bucket_at + timedelta(seconds=interval_seconds) + CLOSE_GRACE <= as_of


BUCKET_LOCK_SQL = """
SELECT pg_advisory_xact_lock(hashtextextended(
  'dashboard_bucket_aggregates:' || %(interval)s::text,
  extract(epoch FROM %(bucket)s::timestamptz)::bigint))
"""

MISSING_BUCKETS_SQL = """
WITH candidates AS (
  SELECT generate_series(%(first)s::timestamptz, %(last)s::timestamptz,
                         %(interval)s::int * interval '1 second') AS bucket_at
)
SELECT c.bucket_at
FROM candidates c
WHERE c.bucket_at + %(interval)s::int * interval '1 second' + %(grace)s::interval
        <= %(as_of)s::timestamptz
  AND NOT EXISTS (
    SELECT 1 FROM benchmarks_v2.dashboard_bucket_fills f
    WHERE f.interval_seconds = %(interval)s AND f.bucket_at = c.bucket_at)
  AND NOT EXISTS (
    SELECT 1 FROM benchmarks_v2.runs r
    WHERE r.status = 'running'
      AND r.scheduled_at >= c.bucket_at
      AND r.scheduled_at < c.bucket_at + %(interval)s::int * interval '1 second')
ORDER BY c.bucket_at
"""

FILL_BUCKET_SQL = """
INSERT INTO benchmarks_v2.dashboard_bucket_aggregates
(provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant,
 value_key, unit, interval_seconds, bucket_at,
 min_value, p25, p50, p75, p90, p95, max_value, value_sum, sample_count, latest_source_at)
SELECT observation.provider, observation.model, observation.benchmark,
       COALESCE(observation.dataset_id, '__all__'),
       evaluation.metric_id, evaluation.metric_version, evaluation.evaluation_variant,
       value.value_key, value.unit, %(interval)s, %(bucket)s,
       MIN(value.value)::float8,
       PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY value.value)::float8,
       PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY value.value)::float8,
       PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY value.value)::float8,
       PERCENTILE_CONT(.9) WITHIN GROUP (ORDER BY value.value)::float8,
       PERCENTILE_CONT(.95) WITHIN GROUP (ORDER BY value.value)::float8,
       MAX(value.value)::float8, SUM(value.value)::float8, COUNT(*)::int,
       MAX(run.scheduled_at)
FROM benchmarks_v2.metric_values value
JOIN benchmarks_v2.metric_evaluations evaluation
  ON evaluation.id = value.metric_evaluation_id
JOIN benchmarks_v2.benchmark_observations observation
  ON observation.id = evaluation.observation_id
JOIN benchmarks_v2.runs run ON run.id = observation.run_id
WHERE observation.status = 'succeeded'
  AND evaluation.status = 'succeeded'
  AND run.status IN ('succeeded', 'partial')
  AND run.scheduled_at >= %(bucket)s
  AND run.scheduled_at < %(bucket)s + %(interval)s::int * interval '1 second'
GROUP BY GROUPING SETS (
  (observation.provider, observation.model, observation.benchmark, observation.dataset_id,
   evaluation.metric_id, evaluation.metric_version, evaluation.evaluation_variant,
   value.value_key, value.unit),
  (observation.provider, observation.model, observation.benchmark,
   evaluation.metric_id, evaluation.metric_version, evaluation.evaluation_variant,
   value.value_key, value.unit)
)
"""


async def missing_buckets(
    pool: AsyncConnectionPool[Any], *, interval_seconds: int, as_of: datetime
) -> list[datetime]:
    """Closed buckets inside the retention window that have never been filled."""
    params = {
        "interval": interval_seconds,
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
    params = {"interval": interval_seconds, "bucket": bucket}
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


async def fill_buckets_covering(
    pool: AsyncConnectionPool[Any], slots: Iterable[datetime], *, as_of: datetime | None = None
) -> int:
    """Refill every closed bucket that contains one of the given run slots."""
    at = as_of or datetime.now(UTC)
    targets: set[tuple[int, datetime]] = set()
    for slot in slots:
        for interval_seconds in BUCKET_INTERVALS:
            bucket = floor_bucket(slot, interval_seconds)
            if is_closed(bucket, interval_seconds, as_of=at):
                targets.add((interval_seconds, bucket))
    for interval_seconds, bucket in sorted(targets):
        await fill_bucket(pool, interval_seconds=interval_seconds, bucket_at=bucket)
    return len(targets)
