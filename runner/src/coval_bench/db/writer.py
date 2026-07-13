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
from datetime import datetime

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.models import Result, Run, RunStatus
from coval_bench.registries import Metric

STATS_MATVIEWS: tuple[str, ...] = ("results_24h", "results_7d", "results_30d")


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
        scheduled_at: datetime | None = None,
    ) -> Run:
        """Insert a ``running`` row into ``benchmarks_v2.runs``.

        Returns a ``Run`` with ``id`` and ``started_at`` populated from the DB.
        """
        sql = """
            INSERT INTO benchmarks_v2.runs
                (runner_sha, dataset_id, dataset_sha256, status, scheduled_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, started_at, finished_at, scheduled_at, runner_sha,
                      dataset_id, dataset_sha256, status, error
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    sql,
                    (runner_sha, dataset_id, dataset_sha256, RunStatus.RUNNING, scheduled_at),
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

        Every ``metric_type`` must be a known ``Metric`` value; an unknown
        value rejects the whole batch before any SQL is executed.
        """
        if not results:
            return

        for r in results:
            try:
                Metric(r.metric_type)
            except ValueError as exc:
                raise ValueError(
                    f"unknown metric_type {r.metric_type!r} (run_id={r.run_id}); "
                    "expected a coval_bench.registries.Metric value"
                ) from exc

        sql = """
            INSERT INTO benchmarks_v2.results
                (run_id, provider, model, voice, benchmark, metric_type,
                 metric_value, metric_units, audio_filename, transcript,
                 status, error, http_version, submit_to_headers_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                r.http_version,
                r.submit_to_headers_ms,
            )
            for r in results
        ]

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(sql, params)
            await conn.commit()

    async def refresh_bucket(self, run_id: int, *, period_seconds: int) -> None:
        """Recompute the series rollup bucket for this run's scheduled_at slot.

        Delete-then-insert the whole bucket from raw result rows in one
        transaction, serialized per bucket by an advisory lock. Recomputing
        the full bucket (not just this run's rows) keeps it correct when runs
        share a slot, and makes the call idempotent. Runs without a
        ``scheduled_at`` are skipped — the migration backfill owns those.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    "SELECT scheduled_at FROM benchmarks_v2.runs WHERE id = %s",
                    (run_id,),
                )
                row = await cur.fetchone()
                bucket_at = row["scheduled_at"] if row is not None else None
                if bucket_at is None:
                    return

                # Serializes concurrent refreshes of one bucket. An empty
                # bucket gives DELETE nothing to lock, so two refreshes can
                # interleave such that the staler recompute commits and the
                # fresher one aborts on the primary key, dropping a run from
                # the slot. Released on commit/abort.
                params = {"bucket": bucket_at, "period": period_seconds}
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended('results_by_bucket',"
                    " extract(epoch FROM %(bucket)s::timestamptz)::bigint))",
                    params,
                )
                # Two statements on purpose: a data-modifying CTE
                # (WITH deleted AS (DELETE ...) INSERT) collides on the primary
                # key — the INSERT cannot see the CTE's deletes.
                await cur.execute(
                    "DELETE FROM benchmarks_v2.results_by_bucket WHERE bucket_at = %(bucket)s",
                    params,
                )
                # Bucket membership: scheduled_at matches exactly, or a legacy
                # null-scheduled row whose created_at falls in
                # [bucket_at, bucket_at + period).
                # dataset_id mirrors migration 20260713_0010: a result's dataset
                # comes from its parent run, except TTS which is always tts-v1.
                await cur.execute(
                    """
                    INSERT INTO benchmarks_v2.results_by_bucket
                        (provider, model, benchmark, dataset_id, metric_type, bucket_at,
                         min_value, p25, p50, p75, max_value, value_sum, sample_count)
                    SELECT r.provider, r.model, r.benchmark,
                           CASE WHEN r.benchmark = 'TTS' THEN 'tts-v1'
                                ELSE rn.dataset_id END,
                           r.metric_type, %(bucket)s,
                           MIN(r.metric_value)::float8,
                           PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY r.metric_value)::float8,
                           PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY r.metric_value)::float8,
                           PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.metric_value)::float8,
                           MAX(r.metric_value)::float8,
                           SUM(r.metric_value)::float8,
                           COUNT(*)::int
                    FROM benchmarks_v2.results r
                    JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
                    WHERE r.status = 'success'
                      AND rn.status IN ('succeeded', 'partial')
                      AND r.metric_value IS NOT NULL
                      AND (
                          rn.scheduled_at = %(bucket)s
                          OR (
                              rn.scheduled_at IS NULL
                              AND r.created_at >= %(bucket)s
                              AND r.created_at < %(bucket)s
                                  + (%(period)s::double precision) * INTERVAL '1 second'
                          )
                      )
                    GROUP BY r.provider, r.model, r.benchmark,
                             CASE WHEN r.benchmark = 'TTS' THEN 'tts-v1'
                                  ELSE rn.dataset_id END,
                             r.metric_type
                    """,
                    params,
                )
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

    async def coval_run_ingested(self, *, provider: str, coval_run_id: str) -> bool:
        """True if a succeeded or partial run already holds rows for this Coval run.

        S2S rows store ``audio_filename = '<coval_run_id>/<sim_id>'``. Lets the
        fetch job skip a re-pulled run so a retry or stale re-pull doesn't
        double-write the day's bucket. Rows from failed runs don't count: they
        never reach the bucket, so a retry must stay free to re-ingest the run.
        """
        sql = """
            SELECT 1
            FROM benchmarks_v2.results r
            JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
            WHERE r.provider = %s
              AND r.benchmark = 'S2S'
              AND split_part(r.audio_filename, '/', 1) = %s
              AND rn.status IN ('succeeded', 'partial')
            LIMIT 1
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (provider, coval_run_id))
                row = await cur.fetchone()
            await conn.commit()
        return row is not None

    async def refresh_stats_matviews(self) -> None:
        """Concurrently refresh the per-window stats materialized views.

        ``CONCURRENTLY`` relies on each view's unique group-key index and does
        not block API reads. Raises on error like the rest of ``RunWriter``.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                for view in STATS_MATVIEWS:
                    await cur.execute(  # noqa: S608 — view names are constants
                        f"REFRESH MATERIALIZED VIEW CONCURRENTLY benchmarks_v2.{view}"
                    )
            await conn.commit()
