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
) -> None:
    """One benchmark run, with a (status, error) result row per entry in *rows*."""
    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        cursor = await conn.execute(
            """
            INSERT INTO benchmarks_v2.runs
                (started_at, runner_sha, dataset_id, dataset_sha256, status)
            VALUES (%(started_at)s, 'deadbeef', 'tts-v1', 'sha', 'succeeded')
            RETURNING id
            """,
            {"started_at": started_at},
        )
        run = await cursor.fetchone()
        assert run is not None
        for status, error in rows:
            await conn.execute(
                """
                INSERT INTO benchmarks_v2.results
                    (run_id, provider, model, benchmark, metric_type, metric_value,
                     metric_units, status, error)
                VALUES (%(run_id)s, %(provider)s, 'some-model', %(benchmark)s, %(metric)s,
                        120.0, 'ms', %(status)s, %(error)s)
                """,
                {
                    "run_id": run["id"],
                    "provider": provider,
                    "benchmark": benchmark,
                    "metric": metric_type,
                    "status": status,
                    "error": error,
                },
            )


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
