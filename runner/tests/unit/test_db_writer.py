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
    Result,
    ResultStatus,
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


def _coval_result(run_id: int, *, benchmark: Benchmark, coval_run_id: str) -> Result:
    return Result(
        run_id=run_id,
        provider="test-provider",
        model="test-model",
        benchmark=benchmark,
        metric_type=Metric.INSTRUCTION_FOLLOWING,
        metric_value=100.0,
        metric_units="percent",
        audio_filename=f"{coval_run_id}/simulation-1",
        status=ResultStatus.SUCCESS,
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
    assert "results_by_bucket" in tables
    assert "llm_turns" in tables

    with pg_conn.cursor() as cur:
        cur.execute("SELECT matviewname FROM pg_matviews WHERE schemaname = 'benchmarks_v2'")
        views = {row[0] for row in cur.fetchall()}
    assert "results_24h" in views

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'benchmarks_v2' AND table_name = 'results'"
        )
        columns = {row[0] for row in cur.fetchall()}
    assert {"http_version", "submit_to_headers_ms"} <= columns

    _downgrade_migrations(pg_conn)

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name = 'benchmarks_v2'"
        )
        schemas = cur.fetchall()
    assert schemas == []


def test_migration_backfills_existing_results(pg_conn: psycopg.Connection[Any]) -> None:
    """Upgrade to 0005, seed results, upgrade to head — the 0006 backfill
    fills the bucket."""
    cfg = _alembic_cfg(_async_dsn(pg_conn))
    alembic_command.upgrade(cfg, "20260611_0005")

    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO benchmarks_v2.runs "
            "(runner_sha, dataset_id, dataset_sha256, status, scheduled_at) "
            "VALUES ('s', 'd', 'h', 'succeeded', now() - interval '1 hour') RETURNING id"
        )
        seed = cur.fetchone()
        assert seed is not None
        run_id = seed[0]
        for value in (1.0, 3.0):
            cur.execute(
                "INSERT INTO benchmarks_v2.results "
                "(run_id, provider, model, benchmark, metric_type, metric_value, "
                " metric_units, status) "
                "VALUES (%s, 'openai', 'whisper-1', 'STT', 'WER', %s, 'ratio', 'success')",
                (run_id, value),
            )

    # Same as `coval-bench db migrate`.
    cfg.attributes["allow_metric_code_cleanup"] = True
    alembic_command.upgrade(cfg, "head")

    with pg_conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            "SELECT dataset_id, min_value, p50, max_value, value_sum, sample_count "
            "FROM benchmarks_v2.results_by_bucket "
            "WHERE provider = 'openai' AND model = 'whisper-1' AND metric_type = 'WER'"
        )
        rows = cur.fetchall()

    # One per-dataset row plus the pooled '__all__' row, identical stats here.
    assert {row["dataset_id"] for row in rows} == {"d", "__all__"}
    for row in rows:
        assert row["sample_count"] == 2
        assert float(row["value_sum"]) == pytest.approx(4.0)
        assert float(row["min_value"]) == pytest.approx(1.0)
        assert float(row["max_value"]) == pytest.approx(3.0)
        assert float(row["p50"]) == pytest.approx(2.0)


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


def test_partial_run(pg_conn: psycopg.Connection[Any]) -> None:
    """finish_run('partial') → status persists."""
    _apply_migrations(pg_conn)

    async def _run() -> int:
        pool = await _make_pool(pg_conn)
        try:
            writer = RunWriter(pool)
            run = await writer.start_run(dataset_id="stt-v1", dataset_sha256="deadbeef")
            assert run.id is not None
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


def test_llm_benchmark_rows_are_accepted(pg_conn: psycopg.Connection[Any]) -> None:
    """Every widened CHECK admits 'LLM', and the migration seeded the Phonely entry."""
    _apply_migrations(pg_conn)
    pg_conn.autocommit = True

    with pg_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO benchmarks_v2.runs (runner_sha, dataset_id, dataset_sha256, status) "
            "VALUES ('sha', 'llm-dental-v1', 'hash', 'succeeded') RETURNING id"
        )
        run_row = cur.fetchone()
        assert run_row is not None
        cur.execute(
            "INSERT INTO benchmarks_v2.results "
            "(run_id, provider, model, benchmark, metric_type, metric_value, metric_units, status) "
            "VALUES (%s, 'phonely', 'phonely-agent', 'LLM', 'TTFT', 0.42, 'seconds', 'success')",
            (run_row[0],),
        )
        cur.execute(
            "INSERT INTO benchmarks_v2.results_by_bucket "
            "(provider, model, benchmark, dataset_id, metric_type, bucket_at, "
            " min_value, p25, p50, p75, max_value, value_sum, sample_count) "
            "VALUES ('phonely', 'phonely-agent', 'LLM', 'llm-dental-v1', 'TTFT', now(), "
            " 0.4, 0.4, 0.42, 0.45, 0.45, 0.85, 2)"
        )
        cur.execute(
            "INSERT INTO benchmarks_v2.models "
            "(modality, provider, model, voice, voices, creator, source, licensing, "
            " on_prem, region, arena_enabled, collected, published, updated_by_user_id) "
            "VALUES ('LLM', 'acme', 'chat-1', NULL, '[]'::jsonb, NULL, 'official-api', "
            " 'proprietary', FALSE, 'us', FALSE, TRUE, FALSE, 'test')"
        )
        cur.execute(
            "SELECT collected, published, arena_enabled, updated_by_user_id "
            "FROM benchmarks_v2.models WHERE modality = 'LLM' AND provider = 'phonely'"
        )
        assert cur.fetchall() == [(True, False, False, "migration:20260901_0025")]


def test_widened_checks_are_validated_and_enforced(pg_conn: psycopg.Connection[Any]) -> None:
    """The NOT VALID swaps end validated, and the re-added CHECKs still reject bad values."""
    _apply_migrations(pg_conn)
    pg_conn.autocommit = True

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT conname, convalidated FROM pg_constraint "
            "WHERE conname IN ('results_benchmark_check', 'results_by_bucket_benchmark_check', "
            " 'benchmark_observations_benchmark_check', "
            " 'models_modality_check', 'model_history_modality_check') "
            "ORDER BY conname"
        )
        rows = cur.fetchall()
    assert len(rows) == 5
    assert all(validated for _, validated in rows), rows

    pg_conn.autocommit = False
    with (
        pytest.raises(psycopg.errors.CheckViolation),
        pg_conn.cursor() as cur,
    ):
        cur.execute(
            "INSERT INTO benchmarks_v2.runs (runner_sha, dataset_id, dataset_sha256, status) "
            "VALUES ('sha', 'ds', 'hash', 'succeeded') RETURNING id"
        )
        run_row = cur.fetchone()
        assert run_row is not None
        cur.execute(
            "INSERT INTO benchmarks_v2.results "
            "(run_id, provider, model, benchmark, metric_type, metric_value, metric_units, status) "
            "VALUES (%s, 'acme', 'x', 'XYZ', 'TTFT', 1.0, 'seconds', 'success')",
            (run_row[0],),
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

            async def add_legacy(
                *,
                name: str,
                coval_run_id: str,
                benchmark: Benchmark = Benchmark.S2S,
                metric_type: str = Metric.INSTRUCTION_FOLLOWING,
                run_status: RunStatus = RunStatus.SUCCEEDED,
            ) -> None:
                run = await writer.start_run(dataset_id=f"legacy-{name}", dataset_sha256="b" * 64)
                assert run.id is not None
                result = _coval_result(run.id, benchmark=benchmark, coval_run_id=coval_run_id)
                if metric_type != result.metric_type:
                    result = result.model_copy(
                        update={"metric_type": metric_type, "metric_units": "seconds"}
                    )
                async with pool.connection() as conn:
                    await conn.execute(
                        """INSERT INTO benchmarks_v2.results
                           (run_id, provider, model, benchmark, metric_type, metric_value,
                            metric_units, audio_filename, status)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            result.run_id,
                            result.provider,
                            result.model,
                            result.benchmark,
                            result.metric_type,
                            result.metric_value,
                            result.metric_units,
                            result.audio_filename,
                            result.status,
                        ),
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
            await add_legacy(name="llm", coval_run_id="RLLM-LEGACY", benchmark=Benchmark.LLM)
            await add_legacy(name="succeeded", coval_run_id="RS2S-SUCCEEDED")
            await add_legacy(
                name="partial", coval_run_id="RS2S-PARTIAL", run_status=RunStatus.PARTIAL
            )
            await add_legacy(
                name="wrong-metric",
                coval_run_id="RS2S-WRONG-METRIC",
                metric_type=Metric.CALL_LENGTH,
            )
            await add_legacy(name="failed", coval_run_id="RS2S-FAILED", run_status=RunStatus.FAILED)
            await add_legacy(
                name="running", coval_run_id="RS2S-RUNNING", run_status=RunStatus.RUNNING
            )

            return {
                "normalized-llm": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RLLM",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                    benchmark=Benchmark.LLM,
                ),
                "legacy-only-llm": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RLLM-LEGACY",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                    benchmark=Benchmark.LLM,
                ),
                "normalized-only-s2s": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RS2S",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "s2s-succeeded": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RS2S-SUCCEEDED",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "s2s-partial": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RS2S-PARTIAL",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "s2s-wrong-metric": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RS2S-WRONG-METRIC",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "s2s-failed": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RS2S-FAILED",
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                ),
                "s2s-running": await writer.coval_metric_ingested(
                    provider="test-provider",
                    coval_run_id="RS2S-RUNNING",
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
        "legacy-only-llm": False,
        "normalized-only-s2s": True,
        "s2s-succeeded": False,
        "s2s-partial": False,
        "s2s-wrong-metric": False,
        "s2s-failed": False,
        "s2s-running": False,
        "failed-eval-succeeded-parent": True,
        "failed-eval-partial-parent": True,
        "failed-parent": False,
        "running-parent": False,
        "wrong-provider": False,
        "wrong-benchmark": False,
        "wrong-prefix": False,
        "different-model-matches": True,
    }
