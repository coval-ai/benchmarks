# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Write-time hourly dashboard sufficient statistics.

Hourly rows are deliberately derived from ``metric_values_by_bucket``.  The
source table is the source of truth; rebuilding an hour deletes and replaces
its complete result set, which makes retries safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import psycopg.rows
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.dashboard_contracts import DEFINITION_REVISION, aggregation_fingerprint
from coval_bench.registries.metrics import TIMELINE_AGGREGATION_RULES

HOURLY_DEFINITION_REVISION = DEFINITION_REVISION

HOUR_LOCK_SQL = """
SELECT pg_advisory_xact_lock(
  hashtextextended('dashboard_hourly_aggregates',
    extract(epoch FROM %(hour)s::timestamptz)::bigint)
)
"""

MARK_HOUR_DIRTY_SQL = """
INSERT INTO benchmarks_v2.dashboard_hourly_state
  (hour_at, dirty, refreshed_at, definition_revision, definition_fingerprint, metadata)
VALUES (%(hour)s, true, NULL, %(definition_revision)s, %(definition_fingerprint)s,
        '{"schema_version": 1}'::jsonb)
ON CONFLICT (hour_at) DO UPDATE SET dirty = true
"""


def floor_hour(value: datetime) -> datetime:
    """Return *value* aligned to the beginning of its UTC hour."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    value = value.astimezone(UTC)
    return value.replace(minute=0, second=0, microsecond=0)


def _rules() -> list[dict[str, Any]]:
    """Serialize the live registry for a parameterized SQL rules CTE."""
    return [
        {
            "metric_type": name,
            "metric_version": rule.version,
            "method": rule.aggregation_method,
            "numerator_keys": list(rule.numerator_keys),
            "denominator_key": rule.denominator_key,
            "scale": rule.ratio_scale,
            "fallback": rule.ratio_fallback,
            "units": {value.key: value.unit for value in rule.values},
        }
        for name, rule in TIMELINE_AGGREGATION_RULES.items()
        if rule.version == "v1"
    ]


# A source bucket is complete only when its primary row and every declared
# ratio operand are present, have matching counts, and use registry units.
_SOURCE_SQL = """
WITH rules AS (
  SELECT * FROM jsonb_to_recordset(%(aggregation_rules)s::jsonb) AS r(
    metric_type text, metric_version text, method text, numerator_keys text[],
    denominator_key text, scale float8, fallback text, units jsonb
  )
), requested_hours AS (
  SELECT unnest(%(hours)s::timestamptz[]) AS hour_at
), source AS (
  SELECT b.provider, b.model, b.benchmark, b.dataset_id, b.metric_type,
         b.metric_version, b.evaluation_variant, b.bucket_at AS source_at,
         h.hour_at,
         r.method, r.fallback, r.scale, r.numerator_keys, r.denominator_key,
         MAX(b.value_sum) FILTER (WHERE b.value_key = 'primary') AS primary_sum,
         MAX(b.sample_count) FILTER (WHERE b.value_key = 'primary') AS sample_count,
         SUM(b.value_sum) FILTER (WHERE b.value_key = ANY(r.numerator_keys)) AS numerator,
         SUM(b.value_sum) FILTER (WHERE b.value_key = r.denominator_key) AS denominator,
         (((r.method = 'mean' AND COUNT(*) = 1)
           OR (r.method = 'ratio' AND COUNT(*) = cardinality(r.numerator_keys) + 2))
          AND MIN(b.sample_count) = MAX(b.sample_count)
          AND BOOL_AND(b.unit = r.units ->> b.value_key)) AS complete
  FROM benchmarks_v2.metric_values_by_bucket b
  JOIN rules r USING (metric_type, metric_version)
  JOIN requested_hours h ON b.bucket_at >= h.hour_at
                         AND b.bucket_at < h.hour_at + interval '1 hour'
  WHERE b.evaluation_variant = 'default'
    AND (b.value_key = 'primary' OR b.value_key = ANY(r.numerator_keys)
         OR b.value_key = r.denominator_key)
  GROUP BY b.provider, b.model, b.benchmark, b.dataset_id, b.metric_type,
           b.metric_version, b.evaluation_variant, b.bucket_at,
           r.method, r.fallback, r.scale, r.numerator_keys, r.denominator_key, h.hour_at
  HAVING COUNT(*) FILTER (WHERE b.value_key = 'primary') = 1
     AND COUNT(*) FILTER (WHERE b.value_key = 'primary'
                          AND b.unit = r.units ->> 'primary') = 1
), grouped AS (
  SELECT provider, model, benchmark, dataset_id, metric_type, metric_version,
         evaluation_variant,
         hour_at,
         SUM(primary_sum)::float8 AS primary_sum,
         SUM(sample_count)::bigint AS sample_count,
         CASE WHEN BOOL_AND(complete) THEN SUM(numerator)::float8 END AS numerator_sum,
         CASE WHEN BOOL_AND(complete) THEN SUM(denominator)::float8 END AS denominator_sum,
         BOOL_AND(complete) AS coverage_complete,
         COUNT(*)::bigint AS source_count, MAX(source_at) AS latest_source_at,
         method, fallback, scale
  FROM source
  GROUP BY provider, model, benchmark, dataset_id, metric_type, metric_version,
           evaluation_variant, hour_at, method, fallback, scale
)
SELECT provider, model, benchmark, dataset_id, metric_type, metric_version,
       evaluation_variant, hour_at, primary_sum, sample_count,
       numerator_sum, denominator_sum, coverage_complete, source_count,
       latest_source_at
FROM grouped
"""


async def refresh_hourly_aggregates(
    pool: AsyncConnectionPool[Any], *, hours: Sequence[datetime]
) -> int:
    """Replace each requested UTC hour and return the number refreshed."""
    normalized = sorted({floor_hour(hour) for hour in hours})
    if not normalized:
        return 0
    fingerprint = aggregation_fingerprint()
    async with pool.connection() as conn:
        async with conn.transaction(), conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            await cur.execute("SET LOCAL statement_timeout = '30s'")
            # Lock every target in deterministic order before reading source
            # buckets. Writers use the same hour-first order, so this snapshot
            # cannot race a source replacement and its dirty marker.
            for hour in normalized:
                await cur.execute(HOUR_LOCK_SQL, {"hour": hour})
            await cur.execute(
                _SOURCE_SQL,
                {"aggregation_rules": Jsonb(_rules()), "hours": normalized},
            )
            rows = await cur.fetchall()
            by_hour: dict[datetime, list[dict[str, Any]]] = {hour: [] for hour in normalized}
            for row in rows:
                by_hour[floor_hour(row["hour_at"])].append(row)
            for hour in normalized:
                params = {
                    "hour": hour,
                    "definition_revision": HOURLY_DEFINITION_REVISION,
                    "definition_fingerprint": fingerprint,
                }
                await cur.execute(
                    "DELETE FROM benchmarks_v2.dashboard_hourly_aggregates "
                    "WHERE hour_at = %(hour)s",
                    {"hour": hour},
                )
                insert_sql = """INSERT INTO benchmarks_v2.dashboard_hourly_aggregates
                            (provider, model, benchmark, dataset_id, metric_type, metric_version,
                             evaluation_variant, hour_at, primary_sum, sample_count, numerator_sum,
                             denominator_sum, coverage_complete, source_count, latest_source_at,
                             definition_revision, metadata)
                            VALUES (%(provider)s, %(model)s, %(benchmark)s, %(dataset_id)s,
                                    %(metric_type)s, %(metric_version)s, %(evaluation_variant)s,
                                    %(hour_at)s, %(primary_sum)s, %(sample_count)s,
                                    %(numerator_sum)s,
                                    %(denominator_sum)s, %(coverage_complete)s, %(source_count)s,
                                    %(latest_source_at)s, %(definition_revision)s,
                                    '{"schema_version": 1}'::jsonb)"""
                await cur.executemany(
                    insert_sql,
                    [
                        {**row, "definition_revision": HOURLY_DEFINITION_REVISION}
                        for row in by_hour[hour]
                    ],
                )
                await cur.execute(
                    """INSERT INTO benchmarks_v2.dashboard_hourly_state
                        (hour_at, dirty, refreshed_at, definition_revision,
                         definition_fingerprint, metadata)
                        VALUES (%(hour)s, false, now(), %(definition_revision)s,
                                %(definition_fingerprint)s, '{"schema_version": 1}'::jsonb)
                        ON CONFLICT (hour_at) DO UPDATE SET dirty = false, refreshed_at = now(),
                          definition_revision = EXCLUDED.definition_revision,
                          definition_fingerprint = EXCLUDED.definition_fingerprint""",
                    params,
                )
        return len(normalized)


async def pending_hourly_aggregates(
    pool: AsyncConnectionPool[Any], *, since: datetime, until: datetime
) -> list[datetime]:
    """List absent, dirty, or definition-stale hours in a bounded interval."""
    if since >= until:
        raise ValueError("since must be earlier than until")
    start = floor_hour(since)
    end = floor_hour(until)
    if until > end:
        end += timedelta(hours=1)
    fingerprint = aggregation_fingerprint()
    sql = """
    WITH requested AS (
      SELECT generate_series(%(since)s::timestamptz, %(until)s::timestamptz - interval '1 hour',
                              interval '1 hour') AS hour_at
    )
    SELECT hour_at FROM requested r
    LEFT JOIN benchmarks_v2.dashboard_hourly_state s USING (hour_at)
    WHERE s.hour_at IS NULL OR s.dirty
       OR s.definition_revision IS DISTINCT FROM %(definition_revision)s
       OR s.definition_fingerprint IS DISTINCT FROM %(definition_fingerprint)s
    UNION
    SELECT s.hour_at FROM benchmarks_v2.dashboard_hourly_state s
    WHERE s.dirty OR s.refreshed_at IS NULL
    ORDER BY hour_at
    """
    async with pool.connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(
            sql,
            {
                "since": start,
                "until": end,
                "definition_revision": HOURLY_DEFINITION_REVISION,
                "definition_fingerprint": fingerprint,
            },
        )
        return [row["hour_at"] for row in await cur.fetchall()]
