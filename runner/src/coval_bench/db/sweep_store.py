# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reads and writes for the sweep tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from psycopg.rows import dict_row
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
    INSERT INTO benchmarks_v2.sweep_queue (sweep_id, sample_id, voice)
    VALUES (%s, %s, %s)
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
      (sweep_id, sample_id, voice, transcript, timings, audio_uri)
    VALUES
      (%(sweep_id)s, %(sample_id)s, %(voice)s, %(transcript)s, %(timings)s, %(audio_uri)s)
    ON CONFLICT (sweep_id, sample_id, voice) DO NOTHING
    RETURNING id
"""

_DEQUEUE = """
    DELETE FROM benchmarks_v2.sweep_queue
    WHERE sweep_id = %(sweep_id)s AND sample_id = %(sample_id)s AND voice = %(voice)s
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


async def open_sweeps(
    pool: Pool, wanted: Sequence[tuple[int, str, str, Sequence[tuple[str, str]]]]
) -> int:
    """Open a sweep, at the model's current version, for each (model id, dataset id,
    dataset version, items) that has none, and queue its items.

    An item is a (sample id, voice) pair; the voice is empty for STT. Returns how many
    were opened.
    """
    opened = 0
    async with pool.connection() as conn, conn.cursor() as cur:
        for model_id, dataset_id, dataset_version, items in wanted:
            params = {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
            }
            async with conn.transaction():
                row = await (await cur.execute(_INSERT_SWEEP, params)).fetchone()
                if row is None:
                    continue
                await cur.executemany(
                    _ENQUEUE, [(row["id"], sample_id, voice) for sample_id, voice in items]
                )
            opened += 1
    return opened


async def unfinished_sweeps(pool: Pool, model_ids: Sequence[int]) -> list[Sweep]:
    """Sweeps at each model's current version that aren't completed, oldest first."""
    async with pool.connection() as conn:
        rows = await (await conn.execute(_UNFINISHED, (list(model_ids),))).fetchall()
    return [Sweep(**row) for row in rows]


async def queued_items(pool: Pool, sweep_id: int) -> set[tuple[str, str]]:
    """The (sample id, voice) items a sweep still has to run and has tries left for."""
    sql = """
        SELECT sample_id, voice FROM benchmarks_v2.sweep_queue
        WHERE sweep_id = %s AND tries_left > 0
    """
    async with pool.connection() as conn:
        rows = await (await conn.execute(sql, (sweep_id,))).fetchall()
    return {(row["sample_id"], row["voice"]) for row in rows}


async def queued_count(pool: Pool, sweep_ids: Sequence[int]) -> int:
    """How many items *sweep_ids* still have queued with tries left."""
    sql = """
        SELECT count(*) AS queued FROM benchmarks_v2.sweep_queue
        WHERE sweep_id = ANY(%s) AND tries_left > 0
    """
    async with pool.connection() as conn:
        row = await (await conn.execute(sql, (list(sweep_ids),))).fetchone()
    return int(row["queued"])


async def store_result(
    pool: Pool,
    *,
    sweep_id: int,
    sample_id: str,
    voice: str,
    transcript: str | None,
    timings: Mapping[str, Any],
    audio_uri: str | None,
) -> int | None:
    """Store an item's result and take the item off the queue.

    Returns the new result's id, or None when the item already had a result, which is kept.
    """
    params = {
        "sweep_id": sweep_id,
        "sample_id": sample_id,
        "voice": voice,
        "transcript": transcript,
        "timings": Jsonb(dict(timings)),
        "audio_uri": audio_uri,
    }
    async with pool.connection() as conn, conn.transaction():
        row = await (await conn.execute(_INSERT_RESULT, params)).fetchone()
        await conn.execute(_DEQUEUE, params)
    return None if row is None else row["id"]


async def record_failure(
    pool: Pool, *, sweep_id: int, sample_id: str, voice: str, error: str
) -> None:
    """Use up one of a queued item's tries and note why."""
    sql = """
        UPDATE benchmarks_v2.sweep_queue SET tries_left = tries_left - 1, last_error = %s
        WHERE sweep_id = %s AND sample_id = %s AND voice = %s AND tries_left > 0
    """
    async with pool.connection() as conn:
        await conn.execute(sql, (error, sweep_id, sample_id, voice))


async def store_score(pool: Pool, result_id: int, score: Score) -> None:
    params = (result_id, score.metric, score.metric_version, score.errors, score.reference_units)
    async with pool.connection() as conn:
        await conn.execute(_INSERT_SCORE, params)


async def complete_sweep(pool: Pool, sweep_id: int) -> bool:
    """Mark a sweep completed if nothing in its queue has tries left. True if it was marked."""
    async with pool.connection() as conn:
        cursor = await conn.execute(_COMPLETE, (sweep_id,))
    return bool(cursor.rowcount)


_QUALITY = """
    WITH scored AS (
        SELECT sw.*, bd.benchmark, bd.condition
        FROM benchmarks_v2.sweeps sw
        JOIN benchmarks_v2.benchmark_datasets bd ON bd.dataset_id = sw.dataset_id
        WHERE sw.completed_at IS NOT NULL
          AND EXISTS (
            SELECT 1 FROM benchmarks_v2.sweep_results r
            JOIN benchmarks_v2.sweep_scores sc ON sc.sweep_result_id = r.id
            WHERE r.sweep_id = sw.id
          )
    ), sizes AS (
        SELECT benchmark, count(*) AS conditions
        FROM benchmarks_v2.benchmark_datasets GROUP BY benchmark
    ), complete AS (
        SELECT DISTINCT ON (s.model_id, s.benchmark)
               s.model_id, s.benchmark, s.model_version, s.dataset_version, z.conditions
        FROM scored s JOIN sizes z USING (benchmark)
        GROUP BY s.model_id, s.benchmark, s.model_version, s.dataset_version, z.conditions
        HAVING count(*) = z.conditions
        ORDER BY s.model_id, s.benchmark, s.model_version DESC, max(s.completed_at) DESC
    )
    SELECT m.id AS model_id, m.modality, m.provider, m.model, s.model_version,
           s.benchmark, b.visibility AS benchmark_visibility,
           s.dataset_version AS benchmark_version, s.condition, c.conditions,
           sc.metric, sc.metric_version,
           sum(sc.errors) AS errors, sum(sc.reference_units) AS reference_units,
           count(*) AS sample_count,
           (SELECT count(*) FROM benchmarks_v2.sweep_results r WHERE r.sweep_id = s.id)
             + (SELECT count(*) FROM benchmarks_v2.sweep_queue q WHERE q.sweep_id = s.id)
             AS expected_sample_count,
           s.completed_at AS measured_at
    FROM scored s
    JOIN complete c USING (model_id, benchmark, model_version, dataset_version)
    JOIN benchmarks_v2.models m ON m.id = s.model_id
    JOIN benchmarks_v2.benchmarks b ON b.slug = s.benchmark
    JOIN benchmarks_v2.sweep_results r ON r.sweep_id = s.id
    JOIN benchmarks_v2.sweep_scores sc ON sc.sweep_result_id = r.id
    GROUP BY m.id, s.id, s.model_version, s.dataset_version, s.completed_at,
             s.benchmark, b.visibility, s.condition, c.conditions, sc.metric, sc.metric_version
    ORDER BY m.id, s.benchmark, s.condition NULLS FIRST, sc.metric, sc.metric_version
"""


async def fetch_quality(pool: Pool) -> list[dict[str, Any]]:
    """Each model's score on each condition of each benchmark, one row per registered dataset.

    A model's rows for a benchmark come from its highest version that has a completed
    sweep, with something scored, of every dataset the benchmark owns at one dataset
    version. So a model keeps its previous scores until the last condition of a new
    version finishes, and has no rows for a benchmark it has never finished.
    ``conditions`` is how many datasets the benchmark owns. A dataset the benchmark
    registry doesn't know is left out. Errors and reference units are summed over every
    scored item, voices included. Raises ``QueryCanceled`` after ten seconds.
    """
    async with (
        pool.connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        await cur.execute("SET LOCAL statement_timeout = '10s'")
        await cur.execute(_QUALITY)
        rows: list[dict[str, Any]] = await cur.fetchall()
        return rows


_LATENCY = """
    SELECT m.id AS model_id, m.modality, m.provider, m.model, v.metric,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY v.value) AS p50,
           percentile_cont(0.9) WITHIN GROUP (ORDER BY v.value) AS p90,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY v.value) AS p95,
           avg(v.value) AS mean, min(v.value) AS min, max(v.value) AS max,
           count(*) AS sample_count
    FROM benchmarks_v2.sweeps s
    JOIN benchmarks_v2.models m ON m.id = s.model_id AND m.version = s.model_version
    JOIN benchmarks_v2.sweep_results r ON r.sweep_id = s.id
    CROSS JOIN LATERAL (
        VALUES ('TTFT', (r.timings->>'ttft_seconds')::float8 * 1000),
               ('TTFS', (r.timings->>'ttfs_seconds')::float8 * 1000),
               ('TTFA', (r.timings->>'ttfa_ms')::float8)
    ) AS v(metric, value)
    WHERE s.completed_at IS NOT NULL AND v.value IS NOT NULL
    GROUP BY m.id, v.metric
"""


async def fetch_sweep_latency(pool: Pool) -> list[dict[str, Any]]:
    """Latency stats per model and metric, from the timings its sweep items recorded.

    They cover the model's completed sweeps at its current version, in milliseconds.
    Raises ``QueryCanceled`` after ten seconds.
    """
    async with (
        pool.connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        await cur.execute("SET LOCAL statement_timeout = '10s'")
        await cur.execute(_LATENCY)
        rows: list[dict[str, Any]] = await cur.fetchall()
        return rows


_SWEEPS = """
    SELECT s.id, s.model_id, m.modality, m.provider, m.model, s.model_version,
           bd.benchmark, bd.condition, s.dataset_id, s.dataset_version,
           s.created_at, s.completed_at,
           (SELECT count(*) FROM benchmarks_v2.sweep_results r WHERE r.sweep_id = s.id)
             AS result_count,
           (SELECT count(*) FROM benchmarks_v2.sweep_queue q
             WHERE q.sweep_id = s.id AND q.tries_left > 0) AS queued_count,
           (SELECT count(*) FROM benchmarks_v2.sweep_queue q
             WHERE q.sweep_id = s.id AND q.tries_left = 0) AS gave_up_count
    FROM benchmarks_v2.sweeps s
    JOIN benchmarks_v2.models m ON m.id = s.model_id
    LEFT JOIN benchmarks_v2.benchmark_datasets bd ON bd.dataset_id = s.dataset_id
    ORDER BY s.created_at, s.id
"""


async def fetch_sweeps(pool: Pool) -> list[dict[str, Any]]:
    """Every sweep, oldest first, with how far through its items it is.

    ``result_count`` is the items that ran, ``queued_count`` the ones still to run and
    ``gave_up_count`` the ones that ran out of tries. Raises ``QueryCanceled`` after ten
    seconds.
    """
    async with (
        pool.connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        await cur.execute("SET LOCAL statement_timeout = '10s'")
        await cur.execute(_SWEEPS)
        rows: list[dict[str, Any]] = await cur.fetchall()
        return rows


_SWEEP_ITEMS = """
    SELECT r.sample_id, NULLIF(r.voice, '') AS voice, 'done' AS status,
           r.transcript, r.timings, r.audio_uri, r.created_at AS ran_at,
           NULL::integer AS tries_left, NULL::text AS last_error,
           COALESCE((
               SELECT jsonb_agg(jsonb_build_object(
                          'metric', sc.metric, 'metric_version', sc.metric_version,
                          'errors', sc.errors, 'reference_units', sc.reference_units)
                        ORDER BY sc.metric, sc.metric_version)
               FROM benchmarks_v2.sweep_scores sc WHERE sc.sweep_result_id = r.id
           ), '[]'::jsonb) AS scores
    FROM benchmarks_v2.sweep_results r
    WHERE r.sweep_id = %(sweep_id)s
    UNION ALL
    SELECT q.sample_id, NULLIF(q.voice, ''),
           CASE WHEN q.tries_left > 0 THEN 'queued' ELSE 'gave_up' END,
           NULL, NULL, NULL, NULL, q.tries_left, q.last_error, '[]'::jsonb
    FROM benchmarks_v2.sweep_queue q
    WHERE q.sweep_id = %(sweep_id)s
    ORDER BY sample_id, voice NULLS FIRST
"""


async def fetch_sweep_items(
    pool: Pool, sweep_id: int, modality: str
) -> list[dict[str, Any]] | None:
    """A sweep's items in sample and voice order, or None if *modality* has no such sweep.

    An item that ran is ``done`` and has its transcript, timings, audio URI and scores. One
    still in the queue is ``queued``, or ``gave_up`` once it has no tries left, and has
    its last error if it has failed.
    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT 1 FROM benchmarks_v2.sweeps sw"
            " JOIN benchmarks_v2.models m ON m.id = sw.model_id"
            " WHERE sw.id = %s AND m.modality = %s",
            (sweep_id, modality),
        )
        if await cur.fetchone() is None:
            return None
        await cur.execute(_SWEEP_ITEMS, {"sweep_id": sweep_id})
        rows: list[dict[str, Any]] = await cur.fetchall()
        return rows


_S2S_SWEEPS = """
    SELECT s.id, s.model_id, m.modality, m.provider, m.model, s.model_version,
           bd.benchmark, bd.condition, s.dataset_id, s.dataset_version,
           s.created_at, s.completed_at,
           (SELECT count(*) FROM benchmarks_v2.s2s_results r WHERE r.sweep_id = s.id)
             AS result_count,
           q.queued_count, q.calling_count, q.scoring_count, q.gave_up_count
    FROM benchmarks_v2.sweeps s
    JOIN benchmarks_v2.models m ON m.id = s.model_id AND m.modality = 'S2S'
    LEFT JOIN benchmarks_v2.benchmark_datasets bd ON bd.dataset_id = s.dataset_id
    CROSS JOIN LATERAL (
        SELECT count(*) FILTER (WHERE q.simulation_id IS NOT NULL) AS scoring_count,
               count(*) FILTER (WHERE q.simulation_id IS NULL AND q.claimed_at IS NOT NULL)
                 AS calling_count,
               count(*) FILTER (WHERE q.claimed_at IS NULL AND q.tries_left > 0)
                 AS queued_count,
               count(*) FILTER (WHERE q.claimed_at IS NULL AND q.tries_left = 0)
                 AS gave_up_count
        FROM benchmarks_v2.s2s_queue q
        WHERE q.sweep_id = s.id
    ) q
    ORDER BY s.created_at, s.id
"""


async def fetch_s2s_sweeps(pool: Pool) -> list[dict[str, Any]]:
    """Every S2S sweep, oldest first, with where its calls are.

    ``result_count`` calls are scored. Of the rest, ``queued_count`` are waiting for a
    worker, ``calling_count`` are claimed and being made, ``scoring_count`` are made and
    waiting on their scores, and ``gave_up_count`` ran out of tries. Raises
    ``QueryCanceled`` after ten seconds.
    """
    async with (
        pool.connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        await cur.execute("SET LOCAL statement_timeout = '10s'")
        await cur.execute(_S2S_SWEEPS)
        rows: list[dict[str, Any]] = await cur.fetchall()
        return rows


_S2S_CALLS = """
    SELECT r.test_case, r.persona, r.repeat, 'done' AS status, r.simulation_id,
           NULL::timestamptz AS claimed_at, r.created_at AS ran_at,
           NULL::integer AS tries_left, NULL::text AS last_error,
           COALESCE((
               SELECT jsonb_agg(jsonb_build_object(
                          'metric', sc.metric, 'metric_version', sc.metric_version,
                          'value', sc.value, 'reference_units', sc.reference_units)
                        ORDER BY sc.metric, sc.metric_version)
               FROM benchmarks_v2.s2s_scores sc WHERE sc.s2s_result_id = r.id
           ), '[]'::jsonb) AS scores
    FROM benchmarks_v2.s2s_results r
    WHERE r.sweep_id = %(sweep_id)s
    UNION ALL
    SELECT q.test_case, q.persona, q.repeat,
           CASE WHEN q.simulation_id IS NOT NULL THEN 'scoring'
                WHEN q.claimed_at IS NOT NULL THEN 'calling'
                WHEN q.tries_left > 0 THEN 'queued'
                ELSE 'gave_up' END,
           q.simulation_id, q.claimed_at, NULL, q.tries_left, q.last_error, '[]'::jsonb
    FROM benchmarks_v2.s2s_queue q
    WHERE q.sweep_id = %(sweep_id)s
    ORDER BY test_case, persona, repeat
"""


async def fetch_s2s_calls(pool: Pool, sweep_id: int) -> list[dict[str, Any]] | None:
    """An S2S sweep's calls in test case, persona and repeat order, or None if there is
    no such S2S sweep.

    A scored call is ``done`` and has its scores. One still in the queue is ``queued``
    while it waits for a worker, ``calling`` once one has claimed it, ``scoring`` once it
    has a simulation id, and ``gave_up`` when it has no tries left.
    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT 1 FROM benchmarks_v2.sweeps sw"
            " JOIN benchmarks_v2.models m ON m.id = sw.model_id"
            " WHERE sw.id = %s AND m.modality = 'S2S'",
            (sweep_id,),
        )
        if await cur.fetchone() is None:
            return None
        await cur.execute(_S2S_CALLS, {"sweep_id": sweep_id})
        rows: list[dict[str, Any]] = await cur.fetchall()
        return rows
