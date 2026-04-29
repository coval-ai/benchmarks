# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.db.

Uses ``pytest-postgresql`` (embedded ``pg_ctl``, no Docker dependency) to spin
up a real Postgres instance.  No remote DB is ever contacted.

The ``pg_proc`` fixture starts a server once per session; each test gets a
clean database via ``DatabaseJanitor`` (created and dropped by the
``postgresql`` client fixture).  Migrations are run inside each test that
needs them via a helper ``_apply_migrations`` that calls Alembic directly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import psycopg
import psycopg.errors
import psycopg.rows
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql, postgresql_proc

from coval_bench.db.conn import get_pool
from coval_bench.db.models import Benchmark, Result, ResultStatus, Run, RunStatus
from coval_bench.db.writer import RunWriter

# ---------------------------------------------------------------------------
# pytest-postgresql fixtures
# ---------------------------------------------------------------------------

pg_proc = postgresql_proc()  # session-scoped Postgres server
pg_conn = postgresql("pg_proc")  # function-scoped clean DB per test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INI_PATH = Path(__file__).parents[2] / "alembic.ini"


def _alembic_cfg(dsn: str) -> AlembicConfig:
    """Return an Alembic config pointed at our alembic.ini with the test DSN."""
    cfg = AlembicConfig(str(_INI_PATH))
    # psycopg3 driver URL for SQLAlchemy
    cfg.set_main_option(
        "sqlalchemy.url",
        dsn.replace("postgresql://", "postgresql+psycopg://"),
    )
    return cfg


def _async_dsn(conn: psycopg.Connection[Any]) -> str:
    """Return a postgresql:// URL suitable for psycopg_pool."""
    info = conn.info
    host = info.host or "localhost"
    port = info.port or 5432
    dbname = info.dbname or "test"
    user = info.user or ""
    password = info.password or ""
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return f"postgresql://{user}@{host}:{port}/{dbname}"


def _apply_migrations(conn: psycopg.Connection[Any]) -> None:
    """Run ``alembic upgrade head`` against the test database."""
    cfg = _alembic_cfg(_async_dsn(conn))
    alembic_command.upgrade(cfg, "head")


def _downgrade_migrations(conn: psycopg.Connection[Any]) -> None:
    """Run ``alembic downgrade base`` against the test database."""
    cfg = _alembic_cfg(_async_dsn(conn))
    alembic_command.downgrade(cfg, "base")


async def _make_pool(conn: psycopg.Connection[Any]) -> AsyncConnectionPool:  # type: ignore[type-arg]
    """Create and open a psycopg3 async pool for the test database."""
    pool: AsyncConnectionPool = AsyncConnectionPool(  # type: ignore[type-arg]
        conninfo=_async_dsn(conn),
        min_size=1,
        max_size=4,
        open=False,
        kwargs={
            "autocommit": False,
            "row_factory": psycopg.rows.dict_row,
        },
    )
    await pool.open()
    return pool


def _make_result(
    run_id: int, *, idx: int = 0, status: ResultStatus = ResultStatus.SUCCESS
) -> Result:
    return Result(
        run_id=run_id,
        provider="openai",
        model="whisper-1",
        voice=None,
        benchmark=Benchmark.STT,
        metric_type="WER",
        metric_value=0.05 + idx * 0.01,
        metric_units="ratio",
        status=status,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_migration_up_down(pg_conn: psycopg.Connection[Any]) -> None:
    """Apply the migration and assert objects exist; downgrade and assert clean."""
    _apply_migrations(pg_conn)
    pg_conn.autocommit = True

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'benchmarks_v2' ORDER BY table_name"
        )
        tables = {row[0] for row in cur.fetchall()}
    assert "runs" in tables
    assert "results" in tables

    with pg_conn.cursor() as cur:
        cur.execute("SELECT matviewname FROM pg_matviews WHERE schemaname = 'benchmarks_v2'")
        views = {row[0] for row in cur.fetchall()}
    assert "results_24h" in views

    _downgrade_migrations(pg_conn)

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name = 'benchmarks_v2'"
        )
        schemas = cur.fetchall()
    assert schemas == []


def test_run_lifecycle(pg_conn: psycopg.Connection[Any]) -> None:
    """start_run → record_result ×3 → finish_run; verify rows and values."""
    _apply_migrations(pg_conn)

    async def _run() -> None:
        pool = await _make_pool(pg_conn)
        try:
            writer = RunWriter(pool)
            run: Run = await writer.start_run(
                runner_sha="abc123",
                dataset_id="stt-v1",
                dataset_sha256="deadbeef",
            )
            assert run.id is not None
            assert run.status == RunStatus.RUNNING
            assert run.started_at is not None

            for i in range(3):
                await writer.record_result(_make_result(run.id, idx=i))

            await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)
        finally:
            await pool.close()

        # Verify via sync psycopg
        pg_conn.autocommit = True
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT status, finished_at FROM benchmarks_v2.runs WHERE id = %s",
                (run.id,),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == "succeeded"
        assert row[1] is not None

        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM benchmarks_v2.results WHERE run_id = %s",
                (run.id,),
            )
            count_row = cur.fetchone()
        assert count_row is not None
        assert count_row[0] == 3

    asyncio.run(_run())


def test_record_results_batch(pg_conn: psycopg.Connection[Any]) -> None:
    """Insert 50 results in one call; assert all rows present."""
    _apply_migrations(pg_conn)

    async def _run() -> None:
        pool = await _make_pool(pg_conn)
        try:
            writer = RunWriter(pool)
            run = await writer.start_run(
                runner_sha="abc123", dataset_id="stt-v1", dataset_sha256="deadbeef"
            )
            assert run.id is not None
            results = [_make_result(run.id, idx=i) for i in range(50)]
            await writer.record_results(results)
            await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)
        finally:
            await pool.close()

        pg_conn.autocommit = True
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM benchmarks_v2.results WHERE run_id = %s",
                (run.id,),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == 50

    asyncio.run(_run())


def test_partial_run(pg_conn: psycopg.Connection[Any]) -> None:
    """1 success + 1 failed result → finish_run('partial') → status persists."""
    _apply_migrations(pg_conn)

    async def _run() -> int:
        pool = await _make_pool(pg_conn)
        try:
            writer = RunWriter(pool)
            run = await writer.start_run(
                runner_sha="abc123", dataset_id="stt-v1", dataset_sha256="deadbeef"
            )
            assert run.id is not None
            await writer.record_results(
                [
                    _make_result(run.id, idx=0, status=ResultStatus.SUCCESS),
                    _make_result(run.id, idx=1, status=ResultStatus.FAILED),
                ]
            )
            await writer.finish_run(run.id, status=RunStatus.PARTIAL)
            return run.id
        finally:
            await pool.close()

    run_id = asyncio.run(_run())

    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute("SELECT status FROM benchmarks_v2.runs WHERE id = %s", (run_id,))
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "partial"


def test_run_with_error(pg_conn: psycopg.Connection[Any]) -> None:
    """finish_run('failed', error=...) → error column persists."""
    _apply_migrations(pg_conn)

    async def _run() -> int:
        pool = await _make_pool(pg_conn)
        try:
            writer = RunWriter(pool)
            run = await writer.start_run(
                runner_sha="abc123", dataset_id="stt-v1", dataset_sha256="deadbeef"
            )
            assert run.id is not None
            await writer.finish_run(
                run.id,
                status=RunStatus.FAILED,
                error="provider X timed out",
            )
            return run.id
        finally:
            await pool.close()

    run_id = asyncio.run(_run())

    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute("SELECT status, error FROM benchmarks_v2.runs WHERE id = %s", (run_id,))
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "failed"
    assert row[1] == "provider X timed out"


def test_results_24h_view(pg_conn: psycopg.Connection[Any]) -> None:
    """Insert 5 results, REFRESH MV, query it, assert aggregates."""
    _apply_migrations(pg_conn)

    async def _insert() -> int:
        pool = await _make_pool(pg_conn)
        try:
            writer = RunWriter(pool)
            run = await writer.start_run(
                runner_sha="abc123", dataset_id="stt-v1", dataset_sha256="deadbeef"
            )
            assert run.id is not None
            results = [
                Result(
                    run_id=run.id,
                    provider="openai",
                    model="whisper-1",
                    benchmark=Benchmark.STT,
                    metric_type="WER",
                    metric_value=float(i) * 0.1,
                    metric_units="ratio",
                    status=ResultStatus.SUCCESS,
                )
                for i in range(1, 6)
            ]
            await writer.record_results(results)
            await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)
            return run.id
        finally:
            await pool.close()

    asyncio.run(_insert())

    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute("REFRESH MATERIALIZED VIEW benchmarks_v2.results_24h")

    with pg_conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            "SELECT avg_value, p50, p95, n "
            "FROM benchmarks_v2.results_24h "
            "WHERE provider = 'openai' AND model = 'whisper-1' AND metric_type = 'WER'"
        )
        row = cur.fetchone()

    assert row is not None
    assert row["n"] == 5
    # avg of [0.1, 0.2, 0.3, 0.4, 0.5] = 0.3
    assert abs(float(row["avg_value"]) - 0.3) < 0.001
    # p50 = median = 0.3
    assert abs(float(row["p50"]) - 0.3) < 0.001


def test_check_constraints(pg_conn: psycopg.Connection[Any]) -> None:
    """Inserting an invalid status must raise CheckViolation."""
    _apply_migrations(pg_conn)

    pg_conn.autocommit = False
    with (
        pytest.raises(psycopg.errors.CheckViolation),
        pg_conn.cursor() as cur,
    ):
        cur.execute(
            """
            INSERT INTO benchmarks_v2.runs
                (runner_sha, dataset_id, dataset_sha256, status)
            VALUES ('sha', 'ds', 'hash', 'invalid_status')
            """
        )
    pg_conn.rollback()


def test_pool_singleton(pg_conn: psycopg.Connection[Any]) -> None:
    """get_pool(settings) returns the same instance on repeated calls."""
    from unittest.mock import MagicMock

    from coval_bench.db import conn as conn_module

    # Reset the module-level singleton so we start clean
    original = conn_module._pool
    conn_module._pool = None
    try:
        settings = MagicMock()
        settings.database_url = _async_dsn(pg_conn)

        async def _check() -> tuple[object, object]:
            p1 = await get_pool(settings)
            p2 = await get_pool(settings)
            return p1, p2

        p1, p2 = asyncio.run(_check())
        assert p1 is p2
    finally:
        # Restore; avoid leaking across tests
        conn_module._pool = original


def test_lifespan_pool(pg_conn: psycopg.Connection[Any]) -> None:
    """lifespan_pool opens and closes the pool correctly."""
    from unittest.mock import MagicMock

    from coval_bench.db import conn as conn_module
    from coval_bench.db.conn import lifespan_pool

    original = conn_module._pool
    conn_module._pool = None
    try:
        settings = MagicMock()
        settings.database_url = _async_dsn(pg_conn)

        async def _use_lifespan() -> bool:
            async with lifespan_pool(settings) as pool:
                return pool.closed is False  # pool is open inside the context

        result = asyncio.run(_use_lifespan())
        assert result
    finally:
        conn_module._pool = original


def test_record_results_batch_rollback_on_failure(pg_conn: psycopg.Connection[Any]) -> None:
    """If one result in a batch violates a constraint, nothing is committed."""
    _apply_migrations(pg_conn)

    async def _run() -> int:
        pool = await _make_pool(pg_conn)
        try:
            writer = RunWriter(pool)
            run = await writer.start_run(
                runner_sha="abc123", dataset_id="stt-v1", dataset_sha256="deadbeef"
            )
            assert run.id is not None
            good = _make_result(run.id, idx=0)
            bad = Result(
                run_id=run.id,
                provider="x",
                model="m",
                benchmark=Benchmark.STT,
                metric_type="WER",
                metric_value=0.1,
                metric_units=None,
                status=ResultStatus.SUCCESS,  # will be overridden in raw SQL below
            )
            # Inject an invalid benchmark value to trigger CheckViolation
            bad = bad.model_copy(update={"benchmark": "INVALID"})  # type: ignore[arg-type]
            with pytest.raises(psycopg.errors.CheckViolation):
                await writer.record_results([good, bad])
            return run.id
        finally:
            await pool.close()

    run_id = asyncio.run(_run())

    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM benchmarks_v2.results WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    assert row is not None
    # Nothing committed — the batch rolled back
    assert row[0] == 0
