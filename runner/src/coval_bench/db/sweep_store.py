# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reads and writes for the sweep tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

Pool = AsyncConnectionPool[Any]


@dataclass(frozen=True)
class Sweep:
    id: int
    model_id: int
    dataset_id: str
    dataset_version: str


_ENSURE = """
    INSERT INTO benchmarks_v2.sweeps (model_id, model_version, dataset_id, dataset_version)
    SELECT m.id, m.version, %(dataset_id)s, %(dataset_version)s
    FROM benchmarks_v2.models m
    WHERE m.id = %(model_id)s
    ON CONFLICT (model_id, model_version, dataset_id, dataset_version) DO NOTHING
"""

_UNFINISHED = """
    SELECT s.id, s.model_id, s.dataset_id, s.dataset_version
    FROM benchmarks_v2.sweeps s
    JOIN benchmarks_v2.models m ON m.id = s.model_id AND m.version = s.model_version
    WHERE s.model_id = ANY(%s) AND s.completed_at IS NULL
    ORDER BY s.created_at, s.id
"""

_INSERT_RESULT = """
    INSERT INTO benchmarks_v2.sweep_results
      (sweep_id, sample_id, transcript, timings, error)
    VALUES
      (%(sweep_id)s, %(sample_id)s, %(transcript)s, %(timings)s, %(error)s)
    ON CONFLICT (sweep_id, sample_id) DO NOTHING
"""


async def ensure_sweeps(pool: Pool, wanted: Sequence[tuple[int, str, str]]) -> int:
    """Open a sweep, at the model's current version, for each (model id, dataset id,
    dataset version) that has none. Returns how many were opened."""
    opened = 0
    async with pool.connection() as conn, conn.cursor() as cur:
        for model_id, dataset_id, dataset_version in wanted:
            await cur.execute(
                _ENSURE,
                {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "dataset_version": dataset_version,
                },
            )
            opened += cur.rowcount
    return opened


async def unfinished_sweeps(pool: Pool, model_ids: Sequence[int]) -> list[Sweep]:
    """Sweeps at each model's current version that aren't completed, oldest first."""
    async with pool.connection() as conn:
        rows = await (await conn.execute(_UNFINISHED, (list(model_ids),))).fetchall()
    return [Sweep(**row) for row in rows]


async def stored_samples(pool: Pool, sweep_id: int) -> set[str]:
    """The sample ids a sweep already has a result for."""
    sql = "SELECT sample_id FROM benchmarks_v2.sweep_results WHERE sweep_id = %s"
    async with pool.connection() as conn:
        rows = await (await conn.execute(sql, (sweep_id,))).fetchall()
    return {row["sample_id"] for row in rows}


async def store_result(
    pool: Pool,
    *,
    sweep_id: int,
    sample_id: str,
    transcript: str | None,
    timings: Mapping[str, Any],
    error: str | None,
) -> None:
    """Store one item's result; *error* set means the call failed. An existing result is kept."""
    params = {
        "sweep_id": sweep_id,
        "sample_id": sample_id,
        "transcript": transcript,
        "timings": Jsonb(dict(timings)),
        "error": error or None,
    }
    async with pool.connection() as conn:
        await conn.execute(_INSERT_RESULT, params)


async def complete_sweep(pool: Pool, sweep_id: int) -> None:
    sql = """
        UPDATE benchmarks_v2.sweeps SET completed_at = now()
        WHERE id = %s AND completed_at IS NULL
    """
    async with pool.connection() as conn:
        await conn.execute(sql, (sweep_id,))
