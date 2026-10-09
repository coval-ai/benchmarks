# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Dashboard rollups: run slots rebuilt on completion, 1h/4h buckets filled once."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

type DashboardPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]

# A 'run' row is one run slot, rebuilt whenever a run in it finishes.
RUN_SLOT = "run"
GRAINS: dict[str, int] = {"1h": 3600, "4h": 14400}
RETENTION = timedelta(days=30)
CLOSE_GRACE = timedelta(minutes=5)
_FILL_STATEMENT_TIMEOUT = "300s"


@dataclass(frozen=True)
class FillResult:
    filled: int
    remaining: int


def grain_for_seconds(seconds: int | None) -> str:
    """The grain the timeline reads for a bucket size; None means run slots."""
    if not seconds:
        return RUN_SLOT
    return next(grain for grain, size in GRAINS.items() if size == seconds)


def floor_rollup(value: datetime, grain: str) -> datetime:
    """Align *value* to the start of its UTC bucket; run slots are already aligned."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    if grain == RUN_SLOT:
        return value.astimezone(UTC)
    epoch = int(value.timestamp())
    return datetime.fromtimestamp(epoch - epoch % GRAINS[grain], tz=UTC)


ROLLUP_LOCK_SQL = """
SELECT pg_advisory_xact_lock(hashtext(%(grain)s),
                             (extract(epoch FROM %(bucket)s::timestamptz) / 3600)::int)
"""

MISSING_ROLLUPS_SQL = """
WITH candidates AS (
  SELECT generate_series(%(first)s::timestamptz, %(last)s::timestamptz, %(step)s) AS bucket_at
)
SELECT c.bucket_at
FROM candidates c
WHERE c.bucket_at + %(step)s + %(grace)s <= %(as_of)s::timestamptz
  AND NOT EXISTS (
    SELECT 1 FROM benchmarks_v2.dashboard_rollup_fills f
    WHERE f.grain = %(grain)s AND f.bucket_at = c.bucket_at)
  AND NOT EXISTS (
    SELECT 1 FROM benchmarks_v2.runs r
    WHERE r.status = 'running'
      AND r.scheduled_at >= c.bucket_at
      AND r.scheduled_at < c.bucket_at + %(step)s)
ORDER BY c.bucket_at
"""

_RUN_SLOT_FILTER = "run.scheduled_at = %(bucket)s"
_INTERVAL_FILTER = "run.scheduled_at >= %(bucket)s AND run.scheduled_at < %(bucket)s + %(step)s"

FILL_ROLLUP_SQL = """
INSERT INTO benchmarks_v2.dashboard_rollups
(provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant,
 value_key, grain, bucket_at,
 min_value, p25, p50, p75, p90, p95, max_value, value_sum, sample_count,
 wer_error_words, wer_reference_words, latest_run_at)
WITH evaluations AS (
  SELECT o.provider, o.model, o.benchmark, o.dataset_id,
         e.metric_id, e.metric_version, e.evaluation_variant,
         e.value, e.roundtrip, e.leading_silence,
         e.substitution_count + e.deletion_count + e.insertion_count AS error_words,
         e.reference_words, run.scheduled_at
  FROM benchmarks_v2.dashboard_metric_values e
  JOIN benchmarks_v2.benchmark_observations o ON o.id = e.observation_id
  JOIN benchmarks_v2.runs run ON run.id = o.run_id
  WHERE o.status = 'succeeded' AND run.status IN ('succeeded', 'partial')
    AND {slot_filter}
), measurements AS (
  SELECT e.*, part.value_key, part.measured
  FROM evaluations e
  CROSS JOIN LATERAL (VALUES
    ('primary', e.value), ('roundtrip', e.roundtrip), ('leading_silence', e.leading_silence)
  ) AS part(value_key, measured)
  WHERE part.measured IS NOT NULL
)
SELECT provider, model, benchmark, COALESCE(dataset_id, '__all__'),
       metric_id, metric_version, evaluation_variant, value_key, %(grain)s, %(bucket)s,
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


async def missing_rollups(
    pool: AsyncConnectionPool[Any], *, grain: str, as_of: datetime
) -> list[datetime]:
    """Closed buckets inside the retention window that have never been filled."""
    params = {
        "grain": grain,
        "step": timedelta(seconds=GRAINS[grain]),
        "first": floor_rollup(as_of - RETENTION, grain),
        "last": floor_rollup(as_of, grain),
        "grace": CLOSE_GRACE,
        "as_of": as_of,
    }
    async with pool.connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(MISSING_ROLLUPS_SQL, params)
        return [row["bucket_at"] for row in await cur.fetchall()]


async def fill_rollup(pool: AsyncConnectionPool[Any], *, grain: str, bucket_at: datetime) -> None:
    """Replace one bucket from raw observations; closed buckets are also recorded as filled."""
    bucket = floor_rollup(bucket_at, grain)
    params = {
        "grain": grain,
        "step": timedelta(seconds=GRAINS.get(grain, 0)),
        "bucket": bucket,
    }
    slot_filter = _RUN_SLOT_FILTER if grain == RUN_SLOT else _INTERVAL_FILTER
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(f"SET LOCAL statement_timeout = '{_FILL_STATEMENT_TIMEOUT}'")
        await conn.execute(ROLLUP_LOCK_SQL, params)
        await conn.execute(
            """DELETE FROM benchmarks_v2.dashboard_rollups
               WHERE grain = %(grain)s AND bucket_at = %(bucket)s""",
            params,
        )
        await conn.execute(FILL_ROLLUP_SQL.format(slot_filter=slot_filter), params)
        if grain != RUN_SLOT:
            await conn.execute(
                """INSERT INTO benchmarks_v2.dashboard_rollup_fills (grain, bucket_at)
                   VALUES (%(grain)s, %(bucket)s)
                   ON CONFLICT (grain, bucket_at) DO UPDATE SET filled_at = now()""",
                params,
            )


async def fill_closed_rollups(
    pool: AsyncConnectionPool[Any], *, as_of: datetime, deadline: float | None = None
) -> FillResult:
    """Fill missing closed buckets oldest-first, committing one at a time."""
    filled = 0
    remaining = 0
    loop = asyncio.get_running_loop()
    for grain in GRAINS:
        pending = await missing_rollups(pool, grain=grain, as_of=as_of)
        for index, bucket_at in enumerate(pending):
            if deadline is not None and loop.time() >= deadline:
                remaining += len(pending) - index
                break
            await fill_rollup(pool, grain=grain, bucket_at=bucket_at)
            filled += 1
    return FillResult(filled, remaining)
