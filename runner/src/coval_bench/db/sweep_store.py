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


@dataclass(frozen=True)
class Score:
    metric: str
    metric_version: str
    errors: int
    reference_units: int


_INSERT_SWEEP = """
    INSERT INTO benchmarks_v2.sweeps (model_id, model_version, dataset_id, dataset_version)
    SELECT m.id, m.version, %(dataset_id)s, %(dataset_version)s
    FROM benchmarks_v2.models m
    WHERE m.id = %(model_id)s
    ON CONFLICT (model_id, model_version, dataset_id, dataset_version) DO NOTHING
    RETURNING id
"""

_ENQUEUE = """
    INSERT INTO benchmarks_v2.sweep_queue (sweep_id, sample_id)
    SELECT %s, unnest(%s::text[])
"""

_UNFINISHED = """
    SELECT s.id, s.model_id, s.dataset_id, s.dataset_version
    FROM benchmarks_v2.sweeps s
    JOIN benchmarks_v2.models m ON m.id = s.model_id AND m.version = s.model_version
    WHERE s.model_id = ANY(%s) AND s.completed_at IS NULL
    ORDER BY s.created_at, s.id
"""

_INSERT_RESULT = """
    INSERT INTO benchmarks_v2.sweep_results (sweep_id, sample_id, transcript, timings)
    VALUES (%(sweep_id)s, %(sample_id)s, %(transcript)s, %(timings)s)
    ON CONFLICT (sweep_id, sample_id) DO NOTHING
    RETURNING id
"""

_DEQUEUE = """
    DELETE FROM benchmarks_v2.sweep_queue
    WHERE sweep_id = %(sweep_id)s AND sample_id = %(sample_id)s
"""

_INSERT_SCORE = """
    INSERT INTO benchmarks_v2.sweep_scores
      (sweep_result_id, metric, metric_version, errors, reference_units)
    VALUES (%s, %s, %s, %s, %s)
"""

_COMPLETE = """
    UPDATE benchmarks_v2.sweeps s SET completed_at = now()
    WHERE s.id = %s AND s.completed_at IS NULL
      AND NOT EXISTS (
        SELECT 1 FROM benchmarks_v2.sweep_queue q WHERE q.sweep_id = s.id AND q.tries_left > 0
      )
"""


async def open_sweeps(pool: Pool, wanted: Sequence[tuple[int, str, str, Sequence[str]]]) -> int:
    """Open a sweep, at the model's current version, for each (model id, dataset id,
    dataset version, sample ids) that has none, and queue its items.

    Returns how many were opened.
    """
    opened = 0
    async with pool.connection() as conn:
        for model_id, dataset_id, dataset_version, sample_ids in wanted:
            params = {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
            }
            async with conn.transaction():
                row = await (await conn.execute(_INSERT_SWEEP, params)).fetchone()
                if row is None:
                    continue
                await conn.execute(_ENQUEUE, (row["id"], list(sample_ids)))
            opened += 1
    return opened


async def unfinished_sweeps(pool: Pool, model_ids: Sequence[int]) -> list[Sweep]:
    """Sweeps at each model's current version that aren't completed, oldest first."""
    async with pool.connection() as conn:
        rows = await (await conn.execute(_UNFINISHED, (list(model_ids),))).fetchall()
    return [Sweep(**row) for row in rows]


async def queued_samples(pool: Pool, sweep_id: int) -> set[str]:
    """The sample ids a sweep still has to run and has tries left for."""
    sql = "SELECT sample_id FROM benchmarks_v2.sweep_queue WHERE sweep_id = %s AND tries_left > 0"
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
) -> int | None:
    """Store an item's result and take the item off the queue.

    Returns the new result's id, or None when the item already had a result, which is kept.
    """
    params = {
        "sweep_id": sweep_id,
        "sample_id": sample_id,
        "transcript": transcript,
        "timings": Jsonb(dict(timings)),
    }
    async with pool.connection() as conn, conn.transaction():
        row = await (await conn.execute(_INSERT_RESULT, params)).fetchone()
        await conn.execute(_DEQUEUE, params)
    return None if row is None else row["id"]


async def record_failure(pool: Pool, *, sweep_id: int, sample_id: str, error: str) -> None:
    """Use up one of a queued item's tries and note why."""
    sql = """
        UPDATE benchmarks_v2.sweep_queue SET tries_left = tries_left - 1, last_error = %s
        WHERE sweep_id = %s AND sample_id = %s AND tries_left > 0
    """
    async with pool.connection() as conn:
        await conn.execute(sql, (error, sweep_id, sample_id))


async def store_score(pool: Pool, result_id: int, score: Score) -> None:
    params = (result_id, score.metric, score.metric_version, score.errors, score.reference_units)
    async with pool.connection() as conn:
        await conn.execute(_INSERT_SCORE, params)


async def complete_sweep(pool: Pool, sweep_id: int) -> bool:
    """Mark a sweep completed if nothing in its queue has tries left. True if it was marked."""
    async with pool.connection() as conn:
        cursor = await conn.execute(_COMPLETE, (sweep_id,))
    return bool(cursor.rowcount)
