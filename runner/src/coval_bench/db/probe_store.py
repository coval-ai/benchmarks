# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reads and writes for ``benchmarks_v2.probe_results``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
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
