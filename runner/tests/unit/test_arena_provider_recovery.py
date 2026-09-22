# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading provider health off the TTS benchmark, against a real Postgres.

Only what the SQL decides is tested here — which run counts, which rows within it, and
what a read failure does. Classification itself is covered in
``test_arena_provider_health.py``, without a database.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql

from coval_bench.arena.provider_health import benchmark_benched_providers

from .conftest import apply_migrations, open_pool

recovery_pg = postgresql("pg_proc")  # shared server from conftest, own per-test DB

# Verbatim from prod: minimax reports errors in websocket frames, with no status code.
DEAD_KEY_ERROR = "task_failed: [1008] insufficient balance"


async def _record_run(
    pool: AsyncConnectionPool[Any],
    *,
    provider: str,
    started_at: datetime,
    rows: list[tuple[str, str | None]],
    metric_type: str = "TTFA",
    benchmark: str = "TTS",
    run_status: str = "succeeded",
    metric_version: str = "v1",
    variant: str = "default",
) -> int:
    """One normalized benchmark run, with one terminal evaluation per entry."""
    async with pool.connection() as conn, conn.transaction():
        conn.row_factory = psycopg.rows.dict_row
        cursor = await conn.execute(
            """
            INSERT INTO benchmarks_v2.runs
                (started_at, runner_sha, dataset_id, dataset_sha256, status)
            VALUES (%(started_at)s, 'deadbeef', 'tts-v1', repeat('a', 64), %(status)s)
            RETURNING id
            """,
            {"started_at": started_at, "status": run_status},
        )
        run = await cursor.fetchone()
        assert run is not None
        for index, (status, error) in enumerate(rows):
            observation = await conn.execute(
                """
                INSERT INTO benchmarks_v2.benchmark_observations
                    (run_id, dataset_id, dataset_sha256, sample_id, provider, model,
                     benchmark, source_kind, status, error, failure_origin)
                VALUES (%(run_id)s, 'tts-v1', repeat('a', 64), %(sample)s, %(provider)s,
                        %(model)s, %(benchmark)s, 'dataset_audio', 'succeeded', NULL, NULL)
                RETURNING id
                """,
                {
                    "run_id": run["id"],
                    "provider": provider,
                    "benchmark": benchmark,
                    "sample": f"sample-{index}",
                    "model": f"model-{index}",
                },
            )
            observation_row = await observation.fetchone()
            assert observation_row is not None
            evaluation = await conn.execute(
                """
                INSERT INTO benchmarks_v2.metric_evaluations
                    (observation_id, metric_id, metric_type, metric_version,
                     evaluation_variant, executor, status)
                VALUES (%(observation_id)s,
                        (SELECT id FROM benchmarks_v2.metrics WHERE code = %(metric)s),
                        %(metric)s, %(version)s, %(variant)s, 'test', 'queued')
                RETURNING id
                """,
                {
                    "observation_id": observation_row["id"],
                    "metric": metric_type,
                    "version": metric_version,
                    "variant": variant,
                },
            )
            evaluation_row = await evaluation.fetchone()
            assert evaluation_row is not None
            if status == "queued":
                continue
            await conn.execute(
                """UPDATE benchmarks_v2.metric_evaluations
                   SET status='running', started_at=%(started)s, updated_at=now()
                   WHERE id=%(id)s""",
                {"id": evaluation_row["id"], "started": started_at},
            )
            if status == "running":
                continue
            if status == "success":
                await conn.execute(
                    """INSERT INTO benchmarks_v2.metric_values
                           (metric_evaluation_id, value_key, unit, value, value_role)
                       VALUES (%(id)s, 'ttfa', 'milliseconds', 120.0, 'primary')""",
                    {"id": evaluation_row["id"]},
                )
                await conn.execute(
                    """UPDATE benchmarks_v2.metric_evaluations
                       SET status='succeeded', finished_at=%(finished)s, updated_at=now()
                       WHERE id=%(id)s""",
                    {"id": evaluation_row["id"], "finished": started_at},
                )
            else:
                await conn.execute(
                    """UPDATE benchmarks_v2.metric_evaluations
                       SET status='failed', finished_at=%(finished)s, updated_at=now(),
                           error=%(error)s
                       WHERE id=%(id)s""",
                    {"id": evaluation_row["id"], "finished": started_at, "error": error},
                )
        return int(run["id"])


def test_a_run_with_only_dead_key_failures_benches_the_provider(
    recovery_pg: psycopg.Connection[Any],
) -> None:
    apply_migrations(recovery_pg)

    async def _run() -> None:
        pool = await open_pool(recovery_pg)
        try:
            now = datetime.now(UTC)
            await _record_run(
                pool,
                provider="minimax",
                started_at=now,
                rows=[("failed", DEAD_KEY_ERROR), ("failed", DEAD_KEY_ERROR)],
            )
            await _record_run(pool, provider="cartesia", started_at=now, rows=[("success", None)])
            assert await benchmark_benched_providers(pool) == frozenset({"minimax"})
        finally:
            await pool.close()

    asyncio.run(_run())


def test_one_success_in_the_run_clears_the_provider(
    recovery_pg: psycopg.Connection[Any],
) -> None:
    """A key that synthesized is a key that works, even if another model of the same
    provider failed on auth. Last-row-wins got this backwards."""
    apply_migrations(recovery_pg)

    async def _run() -> None:
        pool = await open_pool(recovery_pg)
        try:
            await _record_run(
                pool,
                provider="minimax",
                started_at=datetime.now(UTC),
                rows=[("failed", DEAD_KEY_ERROR), ("success", None)],
            )
            assert await benchmark_benched_providers(pool) == frozenset()
        finally:
            await pool.close()

    asyncio.run(_run())


def test_only_the_newest_run_counts(recovery_pg: psycopg.Connection[Any]) -> None:
    """An older success cannot excuse a newer failure, and recovery needs no clearing
    step — the query only ever looks at the latest run."""
    apply_migrations(recovery_pg)

    async def _run() -> None:
        pool = await open_pool(recovery_pg)
        try:
            now = datetime.now(UTC)
            await _record_run(
                pool,
                provider="minimax",
                started_at=now - timedelta(minutes=30),
                rows=[("success", None)],
            )
            await _record_run(
                pool, provider="minimax", started_at=now, rows=[("failed", DEAD_KEY_ERROR)]
            )
            assert await benchmark_benched_providers(pool) == frozenset({"minimax"})

            await _record_run(
                pool,
                provider="minimax",
                started_at=now + timedelta(minutes=30),
                rows=[("success", None)],
            )
            assert await benchmark_benched_providers(pool) == frozenset()
        finally:
            await pool.close()

    asyncio.run(_run())


def test_rows_the_query_must_ignore(recovery_pg: psycopg.Connection[Any]) -> None:
    """STT results, non-TTFA metrics and week-old runs say nothing about a TTS key."""
    apply_migrations(recovery_pg)

    async def _run() -> None:
        pool = await open_pool(recovery_pg)
        try:
            now = datetime.now(UTC)
            await _record_run(
                pool,
                provider="stt-only",
                started_at=now,
                rows=[("failed", DEAD_KEY_ERROR)],
                benchmark="STT",
            )
            await _record_run(
                pool,
                provider="wer-only",
                started_at=now,
                rows=[("failed", DEAD_KEY_ERROR)],
                metric_type="WER",
            )
            await _record_run(
                pool,
                provider="stale",
                started_at=now - timedelta(days=3),
                rows=[("failed", DEAD_KEY_ERROR)],
            )
            assert await benchmark_benched_providers(pool) == frozenset()
        finally:
            await pool.close()

    asyncio.run(_run())


def test_a_read_failure_leaves_the_roster_open(recovery_pg: psycopg.Connection[Any]) -> None:
    """Failing open costs a swapped battle; failing closed would cost every battle."""
    apply_migrations(recovery_pg)

    async def _run() -> None:
        pool = await open_pool(recovery_pg)
        await pool.close()  # the query cannot run
        assert await benchmark_benched_providers(pool) == frozenset()

    asyncio.run(_run())


def test_baseline_outcomes_and_run_id_ties(recovery_pg: psycopg.Connection[Any]) -> None:
    apply_migrations(recovery_pg)

    async def exercise() -> None:
        pool = await open_pool(recovery_pg)
        try:
            now = datetime.now(UTC)
            await _record_run(pool, provider="p", started_at=now, rows=[("success", None)])
            await _record_run(
                pool,
                provider="p",
                started_at=now,
                rows=[("failed", DEAD_KEY_ERROR)],
                run_status="failed",
            )
            assert await benchmark_benched_providers(pool) == frozenset({"p"})
            for overrides in ({"metric_version": "v2"}, {"variant": "reevaluated"}):
                await _record_run(
                    pool, provider="p", started_at=now, rows=[("success", None)], **overrides
                )
            for state in ("queued", "running"):
                await _record_run(pool, provider="p", started_at=now, rows=[(state, None)])
            assert await benchmark_benched_providers(pool) == frozenset({"p"})
            await _record_run(
                pool,
                provider="rate",
                started_at=now,
                rows=[("failed", "HTTP 429 too many requests")],
            )
            assert await benchmark_benched_providers(pool) == frozenset({"p"})
        finally:
            await pool.close()

    asyncio.run(exercise())


def test_parity_sql_covers_failures_and_detects_newer_normalized_run(
    recovery_pg: psycopg.Connection[Any],
) -> None:
    from coval_bench.arena.provider_health_parity import audit_provider_health_parity

    apply_migrations(recovery_pg)

    async def seed(*, recovery: bool = False) -> None:
        pool = await open_pool(recovery_pg)
        try:
            now = datetime.now(UTC)
            if recovery:
                await _record_run(pool, provider="dead", started_at=now, rows=[("success", None)])
                return
            await _record_run(
                pool, provider="dead", started_at=now, rows=[("failed", DEAD_KEY_ERROR)]
            )
            await _record_run(
                pool,
                provider="mixed",
                started_at=now,
                rows=[("failed", DEAD_KEY_ERROR), ("success", None)],
            )
        finally:
            await pool.close()

    asyncio.run(seed())
    recovery_pg.execute("""
        INSERT INTO benchmarks_v2.results
          (run_id, provider, model, benchmark, metric_type, status, error, audio_filename)
        SELECT o.run_id, o.provider, o.model, o.benchmark, e.metric_type,
               CASE WHEN e.status='succeeded' THEN 'success' ELSE 'failed' END,
               e.error, o.sample_id
        FROM benchmarks_v2.benchmark_observations o
        JOIN benchmarks_v2.metric_evaluations e ON e.observation_id=o.id
    """)
    recovery_pg.commit()
    report = audit_provider_health_parity(recovery_pg)
    assert report["complete"] and report["mismatch_count"] == 0
    assert report["legacy"]["dead"]["failed"] == 1
    assert report["normalized"]["mixed"]["succeeded"] == 1
    assert report["legacy"]["dead"]["benched"] is True
    asyncio.run(seed(recovery=True))
    changed = audit_provider_health_parity(recovery_pg)
    assert changed["mismatch_counts"] == {"selected_run_id": 1}
    assert changed["excluded_sets"]["legacy_only"] == ["dead"]
