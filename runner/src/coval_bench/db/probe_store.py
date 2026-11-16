# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reads and writes for ``benchmarks_v2.probe_results``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


@dataclass(frozen=True)
class ProbeResult:
    """One latency probe of one model. Timings are milliseconds."""

    model_id: int
    scheduled_at: datetime
    sent_at: datetime
    voice: str | None = None
    ttft_ms: float | None = None
    ttfs_ms: float | None = None
    ttfa_ms: float | None = None
    error: str | None = None
    failure_origin: Literal["provider", "runner"] | None = None
    status_code: int | None = None

    @property
    def status(self) -> Literal["succeeded", "failed"]:
        return "failed" if self.error is not None else "succeeded"


_INSERT = """
    INSERT INTO benchmarks_v2.probe_results
      (model_id, voice, scheduled_at, sent_at,
       ttft_ms, ttfs_ms, ttfa_ms, status, error, failure_origin, status_code)
    VALUES
      (%(model_id)s, %(voice)s, %(scheduled_at)s, %(sent_at)s,
       %(ttft_ms)s, %(ttfs_ms)s, %(ttfa_ms)s, %(status)s,
       %(error)s, %(failure_origin)s, %(status_code)s)
    ON CONFLICT (model_id, scheduled_at) DO NOTHING
    RETURNING id
"""


async def insert_probe_result(pool: AsyncConnectionPool[Any], result: ProbeResult) -> bool:
    """Store *result*. False when the tick already has a row for the model, which is kept."""
    async with pool.connection() as conn:
        cursor = await conn.execute(_INSERT, {**asdict(result), "status": result.status})
        return await cursor.fetchone() is not None


_HEALTH = """
    SELECT m.id AS model_id, m.modality, m.provider, m.model,
           s.probe_count, s.failed_count, s.last_probe_at, s.last_success_at,
           f.scheduled_at AS failed_at, f.error, f.failure_origin, f.status_code
    FROM (
        SELECT model_id,
               count(*) AS probe_count,
               count(*) FILTER (WHERE status = 'failed') AS failed_count,
               max(scheduled_at) AS last_probe_at,
               max(scheduled_at) FILTER (WHERE status = 'succeeded') AS last_success_at
        FROM benchmarks_v2.probe_results
        WHERE scheduled_at >= now() - interval '24 hours'
        GROUP BY model_id
    ) s
    JOIN benchmarks_v2.models m ON m.id = s.model_id
    LEFT JOIN LATERAL (
        SELECT p.scheduled_at, p.error, p.failure_origin, p.status_code
        FROM benchmarks_v2.probe_results p
        WHERE p.model_id = s.model_id AND p.status = 'failed'
          AND p.scheduled_at >= now() - interval '24 hours'
        ORDER BY p.scheduled_at DESC
        LIMIT 1
    ) f ON true
    ORDER BY s.failed_count DESC, m.id
"""


async def fetch_health(pool: AsyncConnectionPool[Any]) -> list[dict[str, Any]]:
    """Each probed model's last 24 hours: counts, last probe, last success, latest failure."""
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(_HEALTH)
        rows: list[dict[str, Any]] = await cur.fetchall()
        return rows


_LATENCY = """
    SELECT m.id AS model_id, m.modality, m.provider, m.model, v.metric,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY v.value) AS p50,
           percentile_cont(0.9) WITHIN GROUP (ORDER BY v.value) AS p90,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY v.value) AS p95,
           avg(v.value) AS mean, min(v.value) AS min, max(v.value) AS max,
           count(*) AS sample_count
    FROM benchmarks_v2.probe_results p
    JOIN benchmarks_v2.models m ON m.id = p.model_id
    CROSS JOIN LATERAL (
        VALUES ('TTFT', p.ttft_ms), ('TTFS', p.ttfs_ms), ('TTFA', p.ttfa_ms)
    ) AS v(metric, value)
    WHERE p.status = 'succeeded' AND v.value IS NOT NULL AND p.scheduled_at >= %(since)s
    GROUP BY m.id, v.metric
"""

_PROVIDER_FAILURES = """
    SELECT m.id AS model_id, m.modality, m.provider, m.model, count(*) AS failed_count
    FROM benchmarks_v2.probe_results p
    JOIN benchmarks_v2.models m ON m.id = p.model_id
    WHERE p.status = 'failed' AND p.failure_origin = 'provider'
      AND p.scheduled_at >= %(since)s
    GROUP BY m.id
"""


async def fetch_latency(
    pool: AsyncConnectionPool[Any], *, since: datetime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Latency since *since*: stats per model and metric, and provider failures per model.

    Timings are milliseconds. A failure the runner caused is in neither list. Raises
    ``QueryCanceled`` when the read takes longer than ten seconds.
    """
    async with (
        pool.connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        await cur.execute("SET LOCAL statement_timeout = '10s'")
        await cur.execute(_LATENCY, {"since": since})
        stats: list[dict[str, Any]] = await cur.fetchall()
        await cur.execute(_PROVIDER_FAILURES, {"since": since})
        failures: list[dict[str, Any]] = await cur.fetchall()
        return stats, failures


_POINTS = """
    SELECT model_id, provider, model,
           date_bin(%(bucket)s, scheduled_at, TIMESTAMPTZ '2000-01-01 00:00:00+00') AS scheduled_at,
           avg(value) AS value
    FROM (
        SELECT m.id AS model_id, m.provider, m.model, p.scheduled_at, p.status,
               CASE %(metric)s::text
                   WHEN 'TTFT' THEN p.ttft_ms
                   WHEN 'TTFS' THEN p.ttfs_ms
                   ELSE p.ttfa_ms
               END AS value
        FROM benchmarks_v2.probe_results p
        JOIN benchmarks_v2.models m ON m.id = p.model_id
        WHERE m.modality = %(modality)s AND p.scheduled_at >= %(since)s
          AND (p.status = 'succeeded' OR p.failure_origin = 'provider')
    ) probes
    WHERE status = 'failed' OR value IS NOT NULL
    GROUP BY model_id, provider, model, 4
    ORDER BY model_id, 4
"""


async def fetch_latency_points(
    pool: AsyncConnectionPool[Any],
    *,
    modality: str,
    metric: str,
    since: datetime,
    bucket: timedelta,
) -> list[dict[str, Any]]:
    """One point per model and *bucket* since *since*: the mean of *metric* over its probes.

    Buckets start on UTC boundaries. A bucket where every probe failed on the provider's
    side has a null value. A failure the runner caused is left out. Raises
    ``QueryCanceled`` after ten seconds.
    """
    params = {"modality": modality, "metric": metric, "since": since, "bucket": bucket}
    async with (
        pool.connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        await cur.execute("SET LOCAL statement_timeout = '10s'")
        await cur.execute(_POINTS, params)
        rows: list[dict[str, Any]] = await cur.fetchall()
        return rows
