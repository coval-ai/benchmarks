"""Migration-backed checks for dashboard source maintenance boundaries."""

import os
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from pytest_postgresql.factories import postgresql
from structlog.testing import capture_logs

from coval_bench.db import dashboard_source
from coval_bench.db.dashboard_aggregates import repair_dashboard_aggregates
from coval_bench.db.dashboard_source import rebuild_source_bucket
from coval_bench.db.models import RunStatus
from coval_bench.db.writer import RunWriter
from tests.unit import test_normalized_db_writer as storage
from tests.unit.conftest import apply_migrations

pg_conn = postgresql("pg_proc")


def test_aggregation_fingerprint_is_hash_seed_independent() -> None:
    code = (
        "from coval_bench.db.dashboard_contracts import aggregation_fingerprint; "
        "print(aggregation_fingerprint())"
    )
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.append(
            subprocess.check_output(  # noqa: S603
                [sys.executable, "-c", code], env=env, text=True
            ).strip()
        )
    assert outputs[0] == outputs[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error",
    [
        (RunStatus.SUCCEEDED, None),
        (RunStatus.PARTIAL, "cancelled during rollout"),
        (RunStatus.FAILED, "provider failed during rollout"),
    ],
)
async def test_finish_before_dashboard_migration_preserves_run_completion(
    pg_conn: psycopg.Connection[Any], status: RunStatus, error: str | None
) -> None:
    storage._migrate(pg_conn, "20260911_0033")
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run_id, _observation = await storage._observation(writer)
        async with pool.connection() as conn:
            queue = await (
                await conn.execute(
                    "SELECT to_regclass('benchmarks_v2.dashboard_source_refreshes') AS table_name"
                )
            ).fetchone()
        assert queue is not None and queue["table_name"] is None
        with capture_logs() as logs:
            await writer.finish_run(run_id, status=status, error=error)
        async with pool.connection() as conn:
            run = await (
                await conn.execute(
                    "SELECT status, finished_at, error FROM benchmarks_v2.runs WHERE id=%s",
                    (run_id,),
                )
            ).fetchone()
        assert run is not None
        assert run["status"] == str(status)
        assert run["finished_at"] is not None
        assert run["error"] == error
        assert logs == [
            {
                "event": "dashboard_source_refresh_enqueue_skipped",
                "log_level": "warning",
                "run_id": run_id,
                "status": str(status),
                "reason": "dashboard_storage_unavailable",
                "required_migration": "20260914_0034",
            }
        ]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_finish_rolls_back_on_other_enqueue_errors(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run_id, _observation = await storage._observation(writer)
        async with pool.connection() as conn:
            await conn.execute(
                "ALTER TABLE benchmarks_v2.dashboard_source_refreshes "
                "ADD CONSTRAINT reject_test_enqueue CHECK (false) NOT VALID"
            )
        with capture_logs() as logs, pytest.raises(psycopg.errors.CheckViolation):
            await writer.finish_run(run_id, status=RunStatus.FAILED, error="provider failed")
        async with pool.connection() as conn:
            run = await (
                await conn.execute(
                    "SELECT status, finished_at, error FROM benchmarks_v2.runs WHERE id=%s",
                    (run_id,),
                )
            ).fetchone()
            queued = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM benchmarks_v2.dashboard_source_refreshes"
                )
            ).fetchone()
        assert run is not None
        assert run == {"status": "running", "finished_at": None, "error": None}
        assert queued is not None and queued["n"] == 0
        assert logs == []
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_finish_enqueues_source_then_rebuild_marks_hour_dirty(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run_id, observation = await storage._observation(writer)
        evaluation = await storage._evaluation(writer, observation)
        await writer.complete_metric_evaluation(
            storage._required(evaluation.id),
            values=storage._wer_values(storage._required(evaluation.id)),
            finished_at=storage._NOW,
        )
        await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
        async with pool.connection() as conn:
            run = await (
                await conn.execute(
                    "SELECT status, finished_at, error, scheduled_at "
                    "FROM benchmarks_v2.runs WHERE id=%s",
                    (run_id,),
                )
            ).fetchone()
            queued = await (
                await conn.execute("SELECT bucket_at FROM benchmarks_v2.dashboard_source_refreshes")
            ).fetchone()
        assert run is not None
        assert run["status"] == "succeeded" and run["finished_at"] is not None
        assert run["error"] is None
        assert queued is not None and queued["bucket_at"] == run["scheduled_at"]
        bucket = queued["bucket_at"]
        await rebuild_source_bucket(pool, bucket)
        async with pool.connection() as conn:
            state = await (
                await conn.execute(
                    """SELECT dirty, refreshed_at FROM benchmarks_v2.dashboard_hourly_state
                WHERE hour_at=%s""",
                    (bucket.replace(minute=0, second=0, microsecond=0),),
                )
            ).fetchone()
            remaining = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM benchmarks_v2.dashboard_source_refreshes"
                )
            ).fetchone()
        assert state is not None
        assert remaining is not None
        assert state["dirty"] is True and state["refreshed_at"] is None
        assert remaining["n"] == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_hourly_refresh_replaces_rows_and_empty_hour_is_published(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        from coval_bench.db.dashboard_hourly import refresh_hourly_aggregates

        hour = datetime(2026, 9, 14, 12, tzinfo=UTC)
        assert await refresh_hourly_aggregates(pool, hours=[hour]) == 1
        async with pool.connection() as conn:
            state = await (
                await conn.execute(
                    "SELECT dirty, refreshed_at FROM benchmarks_v2.dashboard_hourly_state "
                    "WHERE hour_at=%s",
                    (hour,),
                )
            ).fetchone()
            rows = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM benchmarks_v2.dashboard_hourly_aggregates "
                    "WHERE hour_at=%s",
                    (hour,),
                )
            ).fetchone()
        assert state is not None
        assert rows is not None
        assert state["dirty"] is False and state["refreshed_at"] is not None
        assert rows["n"] == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_failed_source_rebuild_retains_queue_and_previous_source_rows(
    pg_conn: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run_id, observation = await storage._observation(writer)
        evaluation = await storage._evaluation(writer, observation)
        evaluation_id = storage._required(evaluation.id)
        await writer.complete_metric_evaluation(
            evaluation_id, values=storage._wer_values(evaluation_id), finished_at=storage._NOW
        )
        await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
        async with pool.connection() as conn:
            queued_bucket = await (
                await conn.execute("SELECT bucket_at FROM benchmarks_v2.dashboard_source_refreshes")
            ).fetchone()
            assert queued_bucket is not None
            bucket = queued_bucket["bucket_at"]
        await rebuild_source_bucket(pool, bucket)
        async with pool.connection() as conn:
            before = await (
                await conn.execute(
                    """SELECT value_key, value_sum, sample_count
                FROM benchmarks_v2.metric_values_by_bucket WHERE bucket_at=%s
                ORDER BY value_key""",
                    (bucket,),
                )
            ).fetchall()
            await conn.execute(
                """INSERT INTO benchmarks_v2.dashboard_source_refreshes(bucket_at)
                VALUES (%s) ON CONFLICT (bucket_at) DO NOTHING""",
                (bucket,),
            )
        monkeypatch.setattr(dashboard_source, "SOURCE_BUCKET_INSERT_SQL", "SELECT nope")
        with pytest.raises(psycopg.Error):
            await rebuild_source_bucket(pool, bucket)
        async with pool.connection() as conn:
            after = await (
                await conn.execute(
                    """SELECT value_key, value_sum, sample_count
                FROM benchmarks_v2.metric_values_by_bucket WHERE bucket_at=%s
                ORDER BY value_key""",
                    (bucket,),
                )
            ).fetchall()
            queued = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM benchmarks_v2.dashboard_source_refreshes "
                    "WHERE bucket_at=%s",
                    (bucket,),
                )
            ).fetchone()
        assert after == before
        assert queued is not None
        assert queued["n"] == 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_repair_timestamp_move_rebuilds_old_and_new_buckets(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run_id, observation = await storage._observation(writer)
        evaluation = await storage._evaluation(writer, observation)
        evaluation_id = storage._required(evaluation.id)
        await writer.complete_metric_evaluation(
            evaluation_id,
            values=storage._wer_values(evaluation_id),
            finished_at=storage._NOW,
        )
        await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
        await writer.refresh_metric_values_bucket(run_id)
        old = storage._NOW
        new = old.replace(hour=old.hour + 2)
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE benchmarks_v2.runs SET scheduled_at=%s WHERE id=%s", (new, run_id)
            )
            await conn.commit()
        await repair_dashboard_aggregates(pool, buckets=[old, new], as_of=new)
        async with pool.connection() as conn:
            source = await (
                await conn.execute(
                    """SELECT bucket_at, dataset_id, count(*) AS n
                FROM benchmarks_v2.metric_values_by_bucket
                WHERE bucket_at IN (%s, %s)
                GROUP BY bucket_at, dataset_id ORDER BY bucket_at, dataset_id""",
                    (old, new),
                )
            ).fetchall()
            hourly = await (
                await conn.execute(
                    """SELECT hour_at, dataset_id, count(*) AS n
                FROM benchmarks_v2.dashboard_hourly_aggregates
                WHERE hour_at IN (%s, %s)
                GROUP BY hour_at, dataset_id ORDER BY hour_at, dataset_id""",
                    (old, new),
                )
            ).fetchall()
            queued = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM benchmarks_v2.dashboard_source_refreshes"
                )
            ).fetchone()
        assert source == [
            {"bucket_at": new, "dataset_id": "__all__", "n": 4},
            {"bucket_at": new, "dataset_id": "observation-dataset", "n": 4},
        ]
        assert hourly == [
            {"hour_at": new, "dataset_id": "__all__", "n": 1},
            {"hour_at": new, "dataset_id": "observation-dataset", "n": 1},
        ]
        assert queued is not None
        assert queued["n"] == 0
    finally:
        await pool.close()
