# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``RunWriter`` — typed insert helpers for the benchmark persistence layer.

All SQL uses parameterised queries (psycopg ``%s`` style).  No string
interpolation with user data is performed anywhere in this module.

Transaction semantics
---------------------
``record_results`` inserts all rows in a single transaction.  If any single
insert fails (e.g. a check-constraint violation) the entire batch is rolled
back and the exception propagates to the caller.  The orchestrator is
responsible for retry logic.

``record_result`` (singular) delegates to ``record_results`` and shares the
same single-transaction guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.models import Result, Run, RunStatus


class RunWriter:
    """Per-run persistence helper.

    Lifecycle::

        writer = RunWriter(pool)
        run = await writer.start_run(runner_sha=..., dataset_id=..., dataset_sha256=...)
        await writer.record_result(result)
        await writer.record_results([result1, result2, ...])
        await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)

    All methods raise on error; exceptions are never swallowed.
    """

    def __init__(
        self,
        pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]],
    ) -> None:
        self._pool = pool

    async def start_run(
        self,
        *,
        runner_sha: str,
        dataset_id: str,
        dataset_sha256: str,
    ) -> Run:
        """Insert a ``running`` row into ``benchmarks_v2.runs``.

        Returns a ``Run`` with ``id`` and ``started_at`` populated from the DB.
        """
        sql = """
            INSERT INTO benchmarks_v2.runs
                (runner_sha, dataset_id, dataset_sha256, status)
            VALUES (%s, %s, %s, %s)
            RETURNING id, started_at, finished_at, runner_sha,
                      dataset_id, dataset_sha256, status, error
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    sql,
                    (runner_sha, dataset_id, dataset_sha256, RunStatus.RUNNING),
                )
                row = await cur.fetchone()
                if row is None:  # pragma: no cover — unreachable after INSERT RETURNING
                    raise RuntimeError("INSERT INTO runs returned no row")
            await conn.commit()
        return Run.model_validate(dict(row))

    async def record_result(self, result: Result) -> None:
        """Insert a single ``benchmarks_v2.results`` row in its own transaction."""
        await self.record_results([result])

    async def record_results(self, results: Sequence[Result]) -> None:
        """Batch-insert ``results`` in a single transaction.

        All rows are inserted via ``executemany``.  If any row fails (e.g. a
        check-constraint violation), the whole batch is rolled back and the
        exception propagates.  The caller decides whether to retry.
        """
        if not results:
            return

        sql = """
            INSERT INTO benchmarks_v2.results
                (run_id, provider, model, voice, benchmark, metric_type,
                 metric_value, metric_units, audio_filename, transcript,
                 status, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = [
            (
                r.run_id,
                r.provider,
                r.model,
                r.voice,
                r.benchmark,
                r.metric_type,
                r.metric_value,
                r.metric_units,
                r.audio_filename,
                r.transcript,
                r.status,
                r.error,
            )
            for r in results
        ]

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(sql, params)
            await conn.commit()

    async def finish_run(
        self,
        run_id: int,
        *,
        status: RunStatus,
        error: str | None = None,
    ) -> None:
        """Set ``finished_at = now()`` and update ``status`` / ``error`` on a run row."""
        sql = """
            UPDATE benchmarks_v2.runs
            SET finished_at = now(),
                status = %s,
                error  = %s
            WHERE id = %s
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (status, error, run_id))
            await conn.commit()
