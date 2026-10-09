# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Dashboard rollups: a queue of changed run slots, each rebuilt with its 1h and 4h buckets."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
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
RETENTION = timedelta(days=30)
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

CLAIM_SLOTS_SQL = """
DELETE FROM benchmarks_v2.dashboard_rollup_queue
WHERE slot_at IN (
  SELECT slot_at FROM benchmarks_v2.dashboard_rollup_queue ORDER BY slot_at LIMIT 50)
RETURNING slot_at
"""
PRUNE_SQL = {
    table: f"DELETE FROM benchmarks_v2.{table} WHERE {column} < %(before)s"  # noqa: S608
    for table, column in (
        ("dashboard_rollups", "bucket_at"),
        ("dashboard_rollup_queue", "slot_at"),
    )
}


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


async def rebuild_slots(conn: psycopg.AsyncConnection[Any], slots: Sequence[datetime]) -> None:
    """Rebuild the given run slots and, once each, every 1h and 4h bucket containing them."""
    await conn.execute(f"SET LOCAL statement_timeout = '{_REBUILD_STATEMENT_TIMEOUT}'")
    targets = {
        (grain, floor_rollup(slot, grain)) for slot in slots for grain in (RUN_SLOT, *GRAINS)
    }
    for grain, bucket_at in sorted(targets):
        await fill_rollup(conn, grain=grain, bucket_at=bucket_at)


async def drain_rollup_queue(
    pool: AsyncConnectionPool[Any], *, as_of: datetime, deadline: float | None = None
) -> DrainResult:
    """Claim queued slots fifty at a time, rebuild each batch in one commit, then prune."""
    loop = asyncio.get_running_loop()
    rebuilt = 0
    while deadline is None or loop.time() < deadline:
        async with (
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
        ):
            await cur.execute(CLAIM_SLOTS_SQL)
            claimed = [row["slot_at"] for row in await cur.fetchall()]
            if not claimed:
                break
            await rebuild_slots(conn, claimed)
        rebuilt += len(claimed)
    async with pool.connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        async with conn.transaction():
            for sql in PRUNE_SQL.values():
                await cur.execute(sql, {"before": as_of - RETENTION})
        await cur.execute("SELECT count(*) AS n FROM benchmarks_v2.dashboard_rollup_queue")
        row = await cur.fetchone()
    return DrainResult(rebuilt, int(row["n"]) if row else 0)
