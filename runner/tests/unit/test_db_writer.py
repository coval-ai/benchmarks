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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import psycopg.errors
import psycopg.rows
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql

from coval_bench.db.conn import get_pool
from coval_bench.db.models import (
    Benchmark,
    MetricEvaluation,
    MetricExecutor,
    Observation,
    ObservationSourceKind,
    ObservationStatus,
    ProcessingStatus,
    Run,
    RunStatus,
)
from coval_bench.db.writer import RunWriter
from coval_bench.registries import Metric

# ---------------------------------------------------------------------------
# pytest-postgresql fixtures — server shared via conftest ``pg_proc``
# ---------------------------------------------------------------------------

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
    cfg.attributes["allow_metric_code_cleanup"] = True
    alembic_command.upgrade(cfg, "head")


def _downgrade_migrations(conn: psycopg.Connection[Any]) -> None:
    """Run ``alembic downgrade base`` against the test database."""
    cfg = _alembic_cfg(_async_dsn(conn))
    alembic_command.downgrade(cfg, "base")


async def _make_pool(
    conn: psycopg.Connection[Any],
) -> AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]:
    """Create and open a psycopg3 async pool for the test database."""
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]] = AsyncConnectionPool(
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
    assert "llm_turns" in tables
    assert not {"results", "results_by_bucket"} & tables

    with pg_conn.cursor() as cur:
        cur.execute("SELECT matviewname FROM pg_matviews WHERE schemaname = 'benchmarks_v2'")
        views = {row[0] for row in cur.fetchall()}
    assert not {"results_24h", "results_7d", "results_30d"} & views

    _downgrade_migrations(pg_conn)

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name = 'benchmarks_v2'"
        )
        schemas = cur.fetchall()
    assert schemas == []


def test_run_lifecycle(pg_conn: psycopg.Connection[Any]) -> None:
    """start_run → finish_run; verify the run row."""
    _apply_migrations(pg_conn)

    async def _run() -> None:
        pool = await _make_pool(pg_conn)
        try:
            writer = RunWriter(pool)
            run: Run = await writer.start_run(
                dataset_id="stt-v1",
                dataset_sha256="deadbeef",
            )
            assert run.id is not None
            assert run.status == RunStatus.RUNNING
            assert run.started_at is not None

            await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)
        finally:
            await pool.close()

        # Verify via sync psycopg
        pg_conn.autocommit = True
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT status, finished_at, runner_sha FROM benchmarks_v2.runs WHERE id = %s",
                (run.id,),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == "succeeded"
        assert row[1] is not None
        assert row[2] == "untracked"

    asyncio.run(_run())


def test_run_with_error(pg_conn: psycopg.Connection[Any]) -> None:
    """finish_run('failed', error=...) → error column persists."""
    _apply_migrations(pg_conn)

    async def _run() -> int:
        pool = await _make_pool(pg_conn)
        try:
            writer = RunWriter(pool)
            run = await writer.start_run(dataset_id="stt-v1", dataset_sha256="deadbeef")
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


def test_coval_metric_ingestion_reads_normalized_storage(
    pg_conn: psycopg.Connection[Any],
) -> None:
    _apply_migrations(pg_conn)

    async def _run() -> dict[str, bool]:
        pool = await _make_pool(pg_conn)
        try:
            writer = RunWriter(pool)

            async def add_normalized(
                *,
                name: str,
                benchmark: Benchmark = Benchmark.LLM,
                sample_id: str,
                metric_type: str = Metric.INSTRUCTION_FOLLOWING,
                run_status: RunStatus = RunStatus.SUCCEEDED,
                provider: str = "test-provider",
                model: str = "test-model",
                evaluation_status: ProcessingStatus = ProcessingStatus.QUEUED,
            ) -> None:
                run = await writer.start_run(dataset_id=f"dataset-{name}", dataset_sha256="a" * 64)
                assert run.id is not None
                observation = await writer.insert_observation(
                    Observation(
                        run_id=run.id,
                        dataset_id=f"dataset-{name}",
                        dataset_sha256="a" * 64,
                        sample_id=sample_id,
                        provider=provider,
                        model=model,
                        benchmark=benchmark,
                        source_kind=ObservationSourceKind.CONVERSATION_TEXT,
                        status=ObservationStatus.SUCCEEDED,
                    )
                )
                evaluation = await writer.insert_metric_evaluation(
                    MetricEvaluation(
                        observation_id=observation.id,
                        metric_type=metric_type,
                        metric_version="v1",
                        executor=MetricExecutor.INLINE,
                        status=ProcessingStatus.QUEUED,
                    )
                )
                if evaluation_status is ProcessingStatus.FAILED:
                    assert evaluation.id is not None
                    failed_at = datetime.now(UTC)
                    await writer.fail_metric_evaluation_exact(
                        evaluation.id,
                        started_at=failed_at,
                        finished_at=failed_at,
                        error="metric failed",
                    )
                if run_status is not RunStatus.RUNNING:
                    await writer.finish_run(
                        run.id,
                        status=run_status,
                        error="run failed" if run_status is RunStatus.FAILED else None,
                    )

            await add_normalized(name="llm", sample_id="RLLM/sim-1")
            await add_normalized(name="s2s", benchmark=Benchmark.S2S, sample_id="RS2S/sim-1")
            await add_normalized(
                name="failed-eval-succeeded-parent",
                benchmark=Benchmark.S2S,
                sample_id="RFAILED-EVAL/sim-1",
                evaluation_status=ProcessingStatus.FAILED,
            )
            await add_normalized(
                name="failed-eval-partial-parent",
                benchmark=Benchmark.S2S,
                sample_id="RPARTIAL-EVAL/sim-1",
                run_status=RunStatus.PARTIAL,
                evaluation_status=ProcessingStatus.FAILED,
            )
            await add_normalized(
                name="failed-parent",
                benchmark=Benchmark.S2S,
                sample_id="RFAILED-PARENT/sim-1",
                run_status=RunStatus.FAILED,
            )
            await add_normalized(
                name="running-parent",
                benchmark=Benchmark.S2S,
                sample_id="RRUNNING-PARENT/sim-1",
                run_status=RunStatus.RUNNING,
            )
            await add_normalized(
                name="different-model",
                benchmark=Benchmark.S2S,
                sample_id="RMODEL/sim-1",
                model="another-model",
            )

            return {
                "normalized-llm": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RLLM",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                    benchmark=Benchmark.LLM,
                ),
                "normalized-only-s2s": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RS2S",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "failed-eval-succeeded-parent": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RFAILED-EVAL",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "failed-eval-partial-parent": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RPARTIAL-EVAL",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "failed-parent": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RFAILED-PARENT",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "running-parent": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RRUNNING-PARENT",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "wrong-provider": await writer.coval_metric_ingested(
                    provider="other-provider",
                    coval_run_id="RFAILED-EVAL",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "wrong-benchmark": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RFAILED-EVAL",
                    benchmark=Benchmark.LLM,
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "wrong-prefix": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="NO-RFAILED-EVAL",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "different-model-matches": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RMODEL",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
            }
        finally:
            await pool.close()

    result = asyncio.run(_run())
    assert result == {
        "normalized-llm": True,
        "normalized-only-s2s": True,
        "failed-eval-succeeded-parent": True,
        "failed-eval-partial-parent": True,
        "failed-parent": False,
        "running-parent": False,
        "wrong-provider": False,
        "wrong-benchmark": False,
        "wrong-prefix": False,
        "different-model-matches": True,
    }
