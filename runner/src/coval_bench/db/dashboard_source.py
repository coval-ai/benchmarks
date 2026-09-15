# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild normalized source buckets and retain failed maintenance requests."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.dashboard_contracts import DEFINITION_REVISION, aggregation_fingerprint
from coval_bench.db.dashboard_hourly import HOUR_LOCK_SQL, MARK_HOUR_DIRTY_SQL, floor_hour

type DashboardPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]

BUCKET_LOCK_SQL = """
SELECT pg_advisory_xact_lock(hashtextextended('metric_values_by_bucket',
  extract(epoch FROM %(bucket)s::timestamptz)::bigint))
"""

ENQUEUE_RUN_BUCKET_SQL = """
INSERT INTO benchmarks_v2.dashboard_source_refreshes (bucket_at, requested_at)
SELECT scheduled_at, clock_timestamp() FROM benchmarks_v2.runs r
WHERE r.id = %(run_id)s AND r.scheduled_at IS NOT NULL
  AND EXISTS (SELECT 1 FROM benchmarks_v2.benchmark_observations o WHERE o.run_id = r.id)
ON CONFLICT (bucket_at) DO UPDATE SET requested_at = EXCLUDED.requested_at
"""

SOURCE_BUCKET_INSERT_SQL = """
INSERT INTO benchmarks_v2.metric_values_by_bucket
(provider, model, benchmark, dataset_id, metric_type, metric_version,
 evaluation_variant, value_key,
 unit, bucket_at, min_value, p25, p50, p75, max_value, value_sum, sample_count)
SELECT observation.provider, observation.model, observation.benchmark,
       COALESCE(observation.dataset_id, '__all__'), evaluation.metric_type,
       evaluation.metric_version, evaluation.evaluation_variant,
       value.value_key, value.unit, %(bucket)s,
       MIN(value.value)::float8,
       PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY value.value)::float8,
       PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY value.value)::float8,
       PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY value.value)::float8,
       MAX(value.value)::float8, SUM(value.value)::float8, COUNT(*)::int
FROM benchmarks_v2.metric_values value
JOIN benchmarks_v2.metric_evaluations evaluation
  ON evaluation.id = value.metric_evaluation_id
JOIN benchmarks_v2.benchmark_observations observation
  ON observation.id = evaluation.observation_id
JOIN benchmarks_v2.runs run ON run.id = observation.run_id
WHERE observation.status = 'succeeded'
  AND evaluation.status = 'succeeded'
  AND run.status IN ('succeeded', 'partial')
  AND run.scheduled_at = %(bucket)s
GROUP BY GROUPING SETS (
  (observation.provider, observation.model, observation.benchmark,
   observation.dataset_id, evaluation.metric_type,
   evaluation.metric_version, evaluation.evaluation_variant,
   value.value_key, value.unit),
  (observation.provider, observation.model, observation.benchmark,
   evaluation.metric_type, evaluation.metric_version,
   evaluation.evaluation_variant,
   value.value_key, value.unit)
)
"""


def hour_dirty_parameters(bucket: datetime) -> dict[str, Any]:
    """Bind the shared hour marker without duplicating rule identity logic."""
    return {
        "hour": floor_hour(bucket),
        "definition_revision": DEFINITION_REVISION,
        "definition_fingerprint": aggregation_fingerprint(),
    }


def mark_source_hour_dirty(cur: psycopg.Cursor[Any], bucket: datetime) -> None:
    """Lock and mark an hour in a synchronous source-replacement transaction."""
    parameters = hour_dirty_parameters(bucket)
    cur.execute(HOUR_LOCK_SQL, parameters)
    cur.execute(MARK_HOUR_DIRTY_SQL, parameters)


async def rebuild_source_bucket(pool: DashboardPool, bucket: datetime) -> None:
    """Replace a source bucket, clearing its pending request only on commit."""
    parameters = {"bucket": bucket}
    hour_parameters = hour_dirty_parameters(bucket)
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SET LOCAL statement_timeout = '120s'")
        await conn.execute(HOUR_LOCK_SQL, hour_parameters)
        await conn.execute(BUCKET_LOCK_SQL, parameters)
        # Claim the queued row before reading sources. A concurrent finisher
        # re-enqueues after commit, so its later source changes cannot be lost.
        await conn.execute(
            "DELETE FROM benchmarks_v2.dashboard_source_refreshes WHERE bucket_at=%(bucket)s",
            parameters,
        )
        await conn.execute(
            "DELETE FROM benchmarks_v2.metric_values_by_bucket WHERE bucket_at=%(bucket)s",
            parameters,
        )
        await conn.execute(SOURCE_BUCKET_INSERT_SQL, parameters)
        await conn.execute(MARK_HOUR_DIRTY_SQL, hour_parameters)
