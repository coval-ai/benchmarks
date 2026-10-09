# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Dashboard rollups: a queue of changed run slots, each rebuilt with its 1h and 4h buckets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

type DashboardPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]

RUN_SLOT = "run"
GRAINS: dict[str, int] = {"1h": 3600, "4h": 14400}
_GRAIN_BY_SECONDS = {seconds: grain for grain, seconds in GRAINS.items()}
RETENTION: dict[str, timedelta] = {
    RUN_SLOT: timedelta(days=30),
    "1h": timedelta(days=7),
    "4h": timedelta(days=30),
}
_REBUILD_STATEMENT_TIMEOUT = "300s"


@dataclass(frozen=True)
class DrainResult:
    rebuilt: int
    remaining: int


def grain_for_seconds(seconds: int | None) -> str:
    """The grain the timeline reads for a bucket size; None means run slots."""
    return _GRAIN_BY_SECONDS[seconds] if seconds else RUN_SLOT


def floor_rollup(value: datetime, grain: str) -> datetime:
    """Align *value* to the start of its UTC bucket; run slots are already aligned."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    if grain == RUN_SLOT:
        return value.astimezone(UTC)
    epoch = int(value.timestamp())
    return datetime.fromtimestamp(epoch - epoch % GRAINS[grain], tz=UTC)


ENQUEUE_SLOT_SQL = """
INSERT INTO benchmarks_v2.dashboard_rollup_queue (slot_at)
SELECT scheduled_at FROM benchmarks_v2.runs WHERE id = %(run_id)s AND scheduled_at IS NOT NULL
ON CONFLICT (slot_at) DO UPDATE SET queued_at = now()
"""

ROLLUP_LOCK_SQL = """
SELECT pg_advisory_xact_lock(hashtext(%(grain)s),
                             (extract(epoch FROM %(bucket)s::timestamptz) / 3600)::int)
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

PRUNE_ROLLUPS_SQL = """
DELETE FROM benchmarks_v2.dashboard_rollups WHERE grain = %(grain)s AND bucket_at < %(before)s
"""
PRUNE_QUEUE_SQL = "DELETE FROM benchmarks_v2.dashboard_rollup_queue WHERE slot_at < %(before)s"


async def fill_rollup(
    conn: psycopg.AsyncConnection[Any], *, grain: str, bucket_at: datetime
) -> None:
    """Replace one bucket from raw observations inside the caller's transaction."""
    bucket = floor_rollup(bucket_at, grain)
    params = {"grain": grain, "step": timedelta(seconds=GRAINS.get(grain, 0)), "bucket": bucket}
    slot_filter = _RUN_SLOT_FILTER if grain == RUN_SLOT else _INTERVAL_FILTER
    await conn.execute(ROLLUP_LOCK_SQL, params)
    await conn.execute(
        "DELETE FROM benchmarks_v2.dashboard_rollups"
        " WHERE grain = %(grain)s AND bucket_at = %(bucket)s",
        params,
    )
    await conn.execute(FILL_ROLLUP_SQL.format(slot_filter=slot_filter), params)


async def rebuild_slot(conn: psycopg.AsyncConnection[Any], slot_at: datetime) -> None:
    """Rebuild a run slot and the 1h and 4h buckets containing it, then dequeue it."""
    await conn.execute(f"SET LOCAL statement_timeout = '{_REBUILD_STATEMENT_TIMEOUT}'")
    for grain in (RUN_SLOT, *GRAINS):
        await fill_rollup(conn, grain=grain, bucket_at=slot_at)
    await conn.execute(
        "DELETE FROM benchmarks_v2.dashboard_rollup_queue WHERE slot_at = %(slot)s",
        {"slot": slot_at},
    )


async def drain_rollup_queue(
    pool: AsyncConnectionPool[Any], *, as_of: datetime, deadline: float | None = None
) -> DrainResult:
    """Rebuild queued slots oldest-first, one commit each, then drop rows past retention."""
    async with pool.connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(
            "SELECT slot_at FROM benchmarks_v2.dashboard_rollup_queue ORDER BY slot_at"
        )
        queued = [row["slot_at"] for row in await cur.fetchall()]
    loop = asyncio.get_running_loop()
    rebuilt = 0
    for slot_at in queued:
        if deadline is not None and loop.time() >= deadline:
            break
        async with pool.connection() as conn, conn.transaction():
            await rebuild_slot(conn, slot_at)
        rebuilt += 1
    async with pool.connection() as conn, conn.transaction():
        for grain, keep in RETENTION.items():
            await conn.execute(PRUNE_ROLLUPS_SQL, {"grain": grain, "before": as_of - keep})
        await conn.execute(PRUNE_QUEUE_SQL, {"before": as_of - max(RETENTION.values())})
    return DrainResult(rebuilt, len(queued) - rebuilt)
