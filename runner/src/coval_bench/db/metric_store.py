# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``MetricStore``: reads and staff writes of each metric's visibility and collected flag."""

from __future__ import annotations

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from coval_bench.registries import Visibility

MetricPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]

_COLUMNS = "code, display_name, visibility, collected"


class RegisteredMetric(BaseModel):
    code: str
    display_name: str
    visibility: Visibility
    collected: bool


class MetricStore:
    """Per-pool persistence helper for metrics."""

    def __init__(self, pool: MetricPool) -> None:
        self._pool = pool

    async def metrics(self) -> list[RegisteredMetric]:
        sql = f"SELECT {_COLUMNS} FROM benchmarks_v2.metrics ORDER BY code"  # noqa: S608
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql)).fetchall()
        return [RegisteredMetric.model_validate(dict(row)) for row in rows]

    async def update(
        self, code: str, *, visibility: Visibility | None, collected: bool | None
    ) -> tuple[RegisteredMetric, RegisteredMetric] | None:
        """Change the fields that are given.

        Returns the metric before and after, or None when there is no such metric.
        """
        sql = """
            WITH before AS (
                SELECT code, display_name, visibility, collected
                FROM benchmarks_v2.metrics WHERE code = %s
            )
            UPDATE benchmarks_v2.metrics m
            SET visibility = COALESCE(%s, m.visibility), collected = COALESCE(%s, m.collected)
            FROM before
            WHERE m.code = before.code
            RETURNING m.code, m.display_name, m.visibility, m.collected,
                      before.visibility AS was_visibility, before.collected AS was_collected
        """
        async with self._pool.connection() as conn:
            row = await (await conn.execute(sql, (code, visibility, collected))).fetchone()
        if row is None:
            return None
        after = RegisteredMetric.model_validate(dict(row))
        before = after.model_copy(
            update={"visibility": row["was_visibility"], "collected": row["was_collected"]}
        )
        return before, after
