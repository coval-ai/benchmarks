# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the per-run-slot source buckets that feed the 24h dashboard views."""

from __future__ import annotations

from datetime import datetime

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

type DashboardPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]

BUCKET_LOCK_SQL = """
SELECT pg_advisory_xact_lock(hashtextextended('metric_values_by_bucket',
  extract(epoch FROM %(bucket)s::timestamptz)::bigint))
"""

SOURCE_BUCKET_INSERT_SQL = """
INSERT INTO benchmarks_v2.metric_values_by_bucket
(provider, model, benchmark, dataset_id, metric_id, metric_version,
 evaluation_variant, value_key,
 unit, bucket_at, min_value, p25, p50, p75, p90, p95, max_value, value_sum, sample_count)
SELECT observation.provider, observation.model, observation.benchmark,
       COALESCE(observation.dataset_id, '__all__'),
       evaluation.metric_id,
       evaluation.metric_version, evaluation.evaluation_variant,
       value.value_key, value.unit, %(bucket)s,
       MIN(value.value)::float8,
       PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY value.value)::float8,
       PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY value.value)::float8,
       PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY value.value)::float8,
       PERCENTILE_CONT(.9) WITHIN GROUP (ORDER BY value.value)::float8,
       PERCENTILE_CONT(.95) WITHIN GROUP (ORDER BY value.value)::float8,
       MAX(value.value)::float8, SUM(value.value)::float8, COUNT(*)::int
FROM benchmarks_v2.metric_values value
JOIN benchmarks_v2.metric_evaluations evaluation
  ON evaluation.id = value.metric_evaluation_id
JOIN benchmarks_v2.benchmark_observations observation
  ON observation.id = evaluation.observation_id
JOIN benchmarks_v2.runs run ON run.id = observation.run_id
JOIN benchmarks_v2.metrics metric ON metric.id = evaluation.metric_id
WHERE observation.status = 'succeeded'
  AND evaluation.status = 'succeeded'
  AND run.status IN ('succeeded', 'partial')
  AND run.scheduled_at = %(bucket)s
GROUP BY GROUPING SETS (
  (observation.provider, observation.model, observation.benchmark,
   observation.dataset_id,
   evaluation.metric_id,
   evaluation.metric_version, evaluation.evaluation_variant,
   value.value_key, value.unit),
  (observation.provider, observation.model, observation.benchmark,
   evaluation.metric_id,
   evaluation.metric_version,
   evaluation.evaluation_variant,
   value.value_key, value.unit)
)
"""


async def rebuild_source_bucket(pool: DashboardPool, bucket: datetime) -> None:
    """Replace one run slot's source rows from its observations."""
    parameters = {"bucket": bucket}
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SET LOCAL statement_timeout = '120s'")
        await conn.execute(BUCKET_LOCK_SQL, parameters)
        await conn.execute(
            "DELETE FROM benchmarks_v2.metric_values_by_bucket WHERE bucket_at=%(bucket)s",
            parameters,
        )
        await conn.execute(SOURCE_BUCKET_INSERT_SQL, parameters)
