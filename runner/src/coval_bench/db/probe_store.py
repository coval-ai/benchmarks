# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Writes for ``benchmarks_v2.probe_results``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal

from psycopg_pool import AsyncConnectionPool


@dataclass(frozen=True)
class ProbeResult:
    """One latency probe of one model. Timings are milliseconds."""

    benchmark: Literal["STT", "TTS"]
    provider: str
    model: str
    scheduled_at: datetime
    sent_at: datetime
    voice: str | None = None
    ttft_ms: float | None = None
    ttfs_ms: float | None = None
    ttfa_ms: float | None = None
    error: str | None = None
    failure_origin: Literal["provider", "runner"] | None = None

    @property
    def status(self) -> Literal["succeeded", "failed"]:
        return "failed" if self.error is not None else "succeeded"


_INSERT = """
    INSERT INTO benchmarks_v2.probe_results
      (benchmark, provider, model, voice, scheduled_at, sent_at,
       ttft_ms, ttfs_ms, ttfa_ms, status, error, failure_origin)
    VALUES
      (%(benchmark)s, %(provider)s, %(model)s, %(voice)s, %(scheduled_at)s, %(sent_at)s,
       %(ttft_ms)s, %(ttfs_ms)s, %(ttfa_ms)s, %(status)s,
       %(error)s, %(failure_origin)s)
    ON CONFLICT (benchmark, provider, model, scheduled_at) DO NOTHING
    RETURNING id
"""


async def insert_probe_result(pool: AsyncConnectionPool[Any], result: ProbeResult) -> bool:
    """Store *result*. False when the tick already has a row for the model, which is kept."""
    async with pool.connection() as conn:
        cursor = await conn.execute(_INSERT, {**asdict(result), "status": result.status})
        return await cursor.fetchone() is not None
