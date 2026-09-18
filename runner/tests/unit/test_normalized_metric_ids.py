# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Real PostgreSQL identity and lifecycle compatibility checks with fixed SQL."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from pytest_postgresql.factories import postgresql

from coval_bench.db.dashboard_source import rebuild_source_bucket
from coval_bench.db.models import MetricEvaluation, MetricExecutor, ProcessingStatus
from coval_bench.db.writer import RunWriter
from coval_bench.migrations.backfill_normalized_metric_ids import backfill
from tests.unit import test_normalized_db_writer as writer_seed

pg_conn = postgresql("pg_proc")


def _dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    auth = f"{info.user}:{info.password}@" if info.password else f"{info.user}@"
    return (
        f"postgresql://{auth}{info.host or 'localhost'}:{info.port or 5432}/{info.dbname or 'test'}"
    )


def _migrate(conn: psycopg.Connection[Any], revision: str, *, down: bool = False) -> None:
    conn.commit()
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option(
        "sqlalchemy.url", _dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    (command.downgrade if down else command.upgrade)(config, revision)


@pytest.fixture
def historical(pg_conn: Any) -> Any:
    _migrate(pg_conn, "20260914_0035")
    pg_conn.autocommit = True
    with pg_conn.transaction():
        run_id = pg_conn.execute(
            """INSERT INTO benchmarks_v2.runs
               (dataset_id, dataset_sha256, runner_sha, status, finished_at, scheduled_at)
               VALUES ('d', %s, 'test', 'succeeded', now(), date_trunc('hour',now())) RETURNING id""",
            ("a" * 64,),
        ).fetchone()[0]
        observation_id = pg_conn.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (run_id, dataset_id, dataset_sha256, sample_id, provider, model,
                benchmark, source_kind, status)
               VALUES (%s, 'd', %s, 'sample', 'p', 'm', 'STT', 'dataset_audio', 'succeeded')
               RETURNING id""",
            (run_id, "a" * 64),
        ).fetchone()[0]
        for status in ("queued", "running", "succeeded", "failed"):
            evaluation_id = pg_conn.execute(
                """INSERT INTO benchmarks_v2.metric_evaluations
                   (observation_id, metric_type, metric_version, evaluation_variant, executor, status)
                   VALUES (%s, 'WER', 'v1', %s, 'inline', 'queued') RETURNING id""",
                (observation_id, status),
            ).fetchone()[0]
            if status == "queued":
                continue
            pg_conn.execute(
                "UPDATE benchmarks_v2.metric_evaluations SET status='running', started_at=now() WHERE id=%s",
                (evaluation_id,),
            )
            if status == "succeeded":
                pg_conn.execute(
                    """INSERT INTO benchmarks_v2.metric_values
                       (metric_evaluation_id,value_key,unit,value,value_role)
                       VALUES (%s,'primary','percent',9,'primary')""",
                    (evaluation_id,),
                )
            if status in {"succeeded", "failed"}:
                pg_conn.execute(
                    """UPDATE benchmarks_v2.metric_evaluations
                       SET status=%s, finished_at=now(), error=%s WHERE id=%s""",
                    (status, "expected failure" if status == "failed" else None, evaluation_id),
                )
    _migrate(pg_conn, "20260915_0036")
    return pg_conn


def _payloads(conn: Any) -> dict[str, Any]:
    return {
        table: conn.execute(
            f"SELECT to_jsonb(t) - 'metric_id' FROM benchmarks_v2.{table} t ORDER BY 1"
        ).fetchall()
        for table in ("metric_evaluations", "dashboard_metric_values", "metric_values")
    }


def test_all_lifecycle_states_backfill_without_changing_payloads(historical: Any) -> None:
    before = _payloads(historical)
    catalog = historical.execute("SELECT * FROM benchmarks_v2.metrics ORDER BY id").fetchall()
    summary = historical.execute("SELECT * FROM benchmarks_v2.dashboard_summary_state").fetchall()
    result = backfill(historical, apply=True, batch_size=1)
    assert result.status == "completed"
    assert result.updated["metric_evaluations"] == 4
    assert result.updated["dashboard_metric_values"] == 1
    assert _payloads(historical) == before
    # Apply registers definitions added since the seed, never touching existing rows.
    assert (
        historical.execute("SELECT * FROM benchmarks_v2.metrics ORDER BY id").fetchall()[
            : len(catalog)
        ]
        == catalog
    )
    assert (
        historical.execute("SELECT * FROM benchmarks_v2.dashboard_summary_state").fetchall()
        == summary
    )
    assert not any(backfill(historical, apply=True).updated.values())


@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed"])
def test_id_fill_cannot_hide_other_changes(historical: Any, status: str) -> None:
    before = _payloads(historical)
    with pytest.raises(psycopg.errors.RaiseException):
        historical.execute(
            """UPDATE benchmarks_v2.metric_evaluations SET
               metric_id=benchmarks_v2.metric_id_for_code(metric_type), updated_at=updated_at+interval '1 second'
               WHERE status=%s""",
            (status,),
        )
    with pytest.raises(psycopg.errors.CheckViolation, match="same definition"):
        historical.execute(
            """UPDATE benchmarks_v2.metric_evaluations SET
               metric_id=benchmarks_v2.metric_id_for_code('TTFA') WHERE status=%s""",
            (status,),
        )
    assert _payloads(historical) == before


def test_historical_lifecycle_transition_and_parent_consistency(historical: Any) -> None:
    historical.execute(
        """UPDATE benchmarks_v2.metric_evaluations SET status='running', started_at=now()
           WHERE status='queued'"""
    )
    assert historical.execute(
        "SELECT metric_id IS NOT NULL FROM benchmarks_v2.metric_evaluations WHERE evaluation_variant='queued'"
    ).fetchone() == (True,)
    with pytest.raises(psycopg.errors.CheckViolation, match="disagrees with parent"):
        historical.execute(
            """UPDATE benchmarks_v2.dashboard_metric_values SET metric_type='TTFA',
               metric_id=benchmarks_v2.metric_id_for_code('TTFA')"""
        )
    with pytest.raises(psycopg.errors.RaiseException, match="terminal"):
        historical.execute("DELETE FROM benchmarks_v2.metric_evaluations WHERE status='succeeded'")


@pytest.mark.parametrize("hydrate", [False, True])
def test_downgrade_restores_validator_and_retains_rows(historical: Any, hydrate: bool) -> None:
    original = historical.execute(
        "SELECT pg_get_functiondef('benchmarks_v2.validate_metric_transition()'::regprocedure)"
    ).fetchone()
    before = _payloads(historical)
    if hydrate:
        backfill(historical, apply=True)
    _migrate(historical, "20260914_0035", down=True)
    assert _payloads(historical) == before
    assert (
        historical.execute(
            "SELECT pg_get_functiondef('benchmarks_v2.validate_metric_transition()'::regprocedure)"
        ).fetchone()
        == original
    )
    with pytest.raises(psycopg.errors.RaiseException, match="terminal"):
        historical.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET updated_at=now() WHERE status='succeeded'"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed"])
async def test_writer_retry_hydrates_existing_identity(historical: Any, status: str) -> None:
    before = _payloads(historical)
    stored = historical.execute(
        "SELECT id, observation_id FROM benchmarks_v2.metric_evaluations WHERE status=%s",
        (status,),
    ).fetchone()
    request = MetricEvaluation(
        observation_id=stored[1],
        metric_type="WER",
        metric_version="v1",
        evaluation_variant=status,
        executor=MetricExecutor.INLINE,
        status=ProcessingStatus.QUEUED,
    )
    pool = await writer_seed._pool(historical)
    try:
        writer = RunWriter(pool)
        first = await writer.insert_metric_evaluation(request)
        second = await writer.insert_metric_evaluation(request)
        assert first.id == second.id == stored[0]
        assert first.metric_id == second.metric_id
        assert first.metric_id is not None
        assert first.status.value == status
    finally:
        await pool.close()
    assert _payloads(historical) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed"])
async def test_concurrent_historical_retries_preserve_identity_and_payload(
    historical: Any, status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _payloads(historical)
    stored = historical.execute(
        "SELECT id, observation_id FROM benchmarks_v2.metric_evaluations WHERE status=%s",
        (status,),
    ).fetchone()
    request = MetricEvaluation(
        observation_id=stored[1],
        metric_type="WER",
        metric_version="v1",
        evaluation_variant=status,
        executor=MetricExecutor.INLINE,
        status=ProcessingStatus.QUEUED,
    )
    target_id = stored[0]
    barrier = asyncio.Barrier(2)
    connections: set[int] = set()
    original_fetchone = psycopg.AsyncCursor.fetchone

    async def fetchone(cursor: Any) -> Any:
        row = await original_fetchone(cursor)
        if isinstance(row, dict) and row.get("id") == target_id and row.get("metric_id") is None:
            connections.add(id(cursor.connection))
            await barrier.wait()
        return row

    monkeypatch.setattr(psycopg.AsyncCursor, "fetchone", fetchone)
    pool = await writer_seed._pool(historical)
    try:
        writer = RunWriter(pool)
        results = await asyncio.wait_for(
            asyncio.gather(
                writer.insert_metric_evaluation(request),
                writer.insert_metric_evaluation(request),
                return_exceptions=True,
            ),
            timeout=5,
        )
        first, second = results
        assert isinstance(first, MetricEvaluation)
        assert isinstance(second, MetricEvaluation)
        assert first.id == second.id == target_id
        assert first.metric_id == second.metric_id
        assert (
            first.metric_id
            == historical.execute(
                "SELECT id FROM benchmarks_v2.metrics WHERE code='WER'"
            ).fetchone()[0]
        )
        assert first.status.value == second.status.value == status
        assert len(connections) == 2
    finally:
        await pool.close()
    assert _payloads(historical) == before


@pytest.mark.asyncio
async def test_writer_resolves_new_identity_and_rejects_supplied_mismatch(historical: Any) -> None:
    observation_id = historical.execute(
        "SELECT observation_id FROM benchmarks_v2.metric_evaluations LIMIT 1"
    ).fetchone()[0]
    request = MetricEvaluation(
        observation_id=observation_id,
        metric_type="WER",
        metric_version="v1",
        evaluation_variant="fresh",
        executor=MetricExecutor.INLINE,
        status=ProcessingStatus.QUEUED,
    )
    pool = await writer_seed._pool(historical)
    try:
        writer = RunWriter(pool)
        stored, retried = await asyncio.wait_for(
            asyncio.gather(
                writer.insert_metric_evaluation(request),
                writer.insert_metric_evaluation(request),
            ),
            timeout=5,
        )
        assert stored.id == retried.id
        assert stored.metric_id == retried.metric_id
        assert (
            stored.metric_id
            == historical.execute(
                "SELECT id FROM benchmarks_v2.metrics WHERE code='WER'"
            ).fetchone()[0]
        )
        wrong_id = historical.execute(
            "SELECT id FROM benchmarks_v2.metrics WHERE code='TTFA'"
        ).fetchone()[0]
        with pytest.raises(ValueError, match="same definition"):
            await writer.insert_metric_evaluation(
                request.model_copy(update={"metric_id": wrong_id})
            )
        assert (await writer.insert_metric_evaluation(request)).id == stored.id
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_mixed_historical_and_new_ids_share_one_source_group(historical: Any) -> None:
    with historical.transaction():
        observation_id = historical.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (run_id,dataset_id,dataset_sha256,sample_id,provider,model,benchmark,source_kind,status)
               SELECT run_id,dataset_id,dataset_sha256,'second',provider,model,benchmark,source_kind,status
               FROM benchmarks_v2.benchmark_observations LIMIT 1 RETURNING id"""
        ).fetchone()[0]
        evaluation_id = historical.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (observation_id,metric_type,metric_version,evaluation_variant,executor,status)
               VALUES (%s,'WER','v1','succeeded','inline','queued') RETURNING id""",
            (observation_id,),
        ).fetchone()[0]
        historical.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET status='running',started_at=now() WHERE id=%s",
            (evaluation_id,),
        )
        historical.execute(
            """INSERT INTO benchmarks_v2.metric_values
               (metric_evaluation_id,value_key,unit,value,value_role)
               VALUES (%s,'primary','percent',3,'primary')""",
            (evaluation_id,),
        )
        historical.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET status='succeeded',finished_at=now() WHERE id=%s",
            (evaluation_id,),
        )
    bucket = historical.execute("SELECT scheduled_at FROM benchmarks_v2.runs LIMIT 1").fetchone()[0]
    pool = await writer_seed._pool(historical)
    try:
        await rebuild_source_bucket(pool, bucket)
    finally:
        await pool.close()
    assert historical.execute(
        """SELECT dataset_id,sample_count,value_sum,metric_id IS NOT NULL
           FROM benchmarks_v2.metric_values_by_bucket ORDER BY dataset_id"""
    ).fetchall() == [("__all__", 2, 12, True), ("d", 2, 12, True)]
    assert historical.execute(
        "SELECT count(*) FROM benchmarks_v2.metric_evaluations WHERE metric_id IS NULL"
    ).fetchone() == (4,)


def test_0036_adds_compatible_normalized_id_columns(pg_conn: psycopg.Connection[Any]) -> None:
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option(
        "sqlalchemy.url", _dsn(pg_conn).replace("postgresql://", "postgresql+psycopg://")
    )
    command.upgrade(config, "20260915_0036")
    with pg_conn.cursor() as cur:
        cur.execute(
            """SELECT table_name, is_nullable FROM information_schema.columns
               WHERE table_schema='benchmarks_v2' AND column_name='metric_id'
               ORDER BY table_name"""
        )
        assert cur.fetchall() == [
            ("dashboard_hourly_aggregates", "NO"),
            ("dashboard_metric_values", "YES"),
            ("metric_evaluations", "YES"),
            ("metric_values_by_bucket", "YES"),
        ]
        cur.execute(
            "SELECT definition_revision FROM benchmarks_v2.dashboard_summary_state WHERE id=true"
        )
        assert cur.fetchone() is not None
