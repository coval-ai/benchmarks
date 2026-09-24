"""Migration-backed checks for dashboard source maintenance boundaries."""

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from pytest_postgresql.factories import postgresql
from structlog.testing import capture_logs

from coval_bench.db import dashboard_aggregates, dashboard_hourly, dashboard_source
from coval_bench.db.dashboard_aggregates import MaintenanceResult, repair_dashboard_aggregates
from coval_bench.db.dashboard_contracts import DEFINITION_REVISION, aggregation_fingerprint
from coval_bench.db.dashboard_hourly import PendingHourlyStatus, floor_hour
from coval_bench.db.dashboard_source import rebuild_source_bucket
from coval_bench.db.dashboard_summaries import RefreshResult
from coval_bench.db.models import RunStatus
from coval_bench.db.writer import RunWriter
from tests.unit import test_normalized_db_writer as storage
from tests.unit.conftest import apply_migrations

from .test_dashboard_hourly import _bucket

pg_conn = postgresql("pg_proc")


@pytest.mark.asyncio
async def test_reconciliation_gives_summary_the_remaining_maintenance_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = MagicMock()
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall.return_value = []
    conn.execute.return_value = cursor
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = conn
    pool.connection.return_value = connection_context

    monkeypatch.setattr(
        dashboard_aggregates, "pending_hourly_aggregates", AsyncMock(return_value=[])
    )
    refresh = AsyncMock(return_value=RefreshResult("published", 7))
    monkeypatch.setattr(dashboard_aggregates, "refresh_summary_snapshots", refresh)

    deadlines: list[float] = []
    real_timeout_at = asyncio.timeout_at

    def recording_timeout_at(deadline: float) -> asyncio.Timeout:
        deadlines.append(deadline)
        return real_timeout_at(deadline)

    monkeypatch.setattr(asyncio, "timeout_at", recording_timeout_at)
    loop = asyncio.get_running_loop()
    before = loop.time()
    result = await dashboard_aggregates.reconcile_dashboard_aggregates(
        pool, as_of=datetime(2026, 9, 16, tzinfo=UTC)
    )
    after = loop.time()

    assert result == MaintenanceResult(0, 0, RefreshResult("published", 7))
    assert len(deadlines) == 1
    assert deadlines[0] - before >= dashboard_aggregates._MAINTENANCE_TIMEOUT_SECONDS - 0.1
    assert deadlines[0] - after <= dashboard_aggregates._MAINTENANCE_TIMEOUT_SECONDS
    refresh.assert_awaited_once_with(pool, as_of=datetime(2026, 9, 16, tzinfo=UTC))


@pytest.mark.asyncio
async def test_reconciliation_registers_catalog_once_and_commits_hours_individually(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = MagicMock()
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall.return_value = []
    conn.execute.return_value = cursor
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = conn
    pool.connection.return_value = connection_context

    first = datetime(2026, 9, 16, 10, tzinfo=UTC)
    second = first + timedelta(hours=1)
    monkeypatch.setattr(
        dashboard_aggregates,
        "pending_hourly_aggregates",
        AsyncMock(return_value=[first, second]),
    )
    monkeypatch.setattr(
        dashboard_aggregates,
        "pending_hourly_status",
        AsyncMock(return_value=PendingHourlyStatus(0, None)),
    )
    catalog = AsyncMock(return_value={"TTFT": 1})
    writer = AsyncMock()
    monkeypatch.setattr(dashboard_aggregates, "prepare_hourly_aggregate_catalog", catalog)
    monkeypatch.setattr(dashboard_aggregates, "_refresh_hourly_aggregate", writer)
    summary = AsyncMock(return_value=RefreshResult("published", 3))
    monkeypatch.setattr(dashboard_aggregates, "refresh_summary_snapshots", summary)

    result = await dashboard_aggregates.reconcile_dashboard_aggregates(
        pool, as_of=datetime(2026, 9, 16, 12, tzinfo=UTC)
    )

    assert result.selected == 2
    assert result.committed == 2
    assert result.remaining == 0
    catalog.assert_awaited_once_with(pool)
    assert [call.args[1] for call in writer.await_args_list] == [first, second]
    summary.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconciliation_does_not_treat_arbitrary_timeout_as_phase_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = MagicMock()
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall.return_value = []
    conn.execute.return_value = cursor
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = conn
    pool.connection.return_value = connection_context

    hour = datetime(2026, 9, 16, 10, tzinfo=UTC)
    monkeypatch.setattr(
        dashboard_aggregates,
        "pending_hourly_aggregates",
        AsyncMock(return_value=[hour]),
    )
    monkeypatch.setattr(
        dashboard_aggregates,
        "pending_hourly_status",
        AsyncMock(return_value=PendingHourlyStatus(1, hour)),
    )
    monkeypatch.setattr(
        dashboard_aggregates, "prepare_hourly_aggregate_catalog", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        dashboard_aggregates,
        "_refresh_hourly_aggregate",
        AsyncMock(side_effect=TimeoutError("statement timeout")),
    )
    summary = AsyncMock(return_value=RefreshResult("published", 4))
    monkeypatch.setattr(dashboard_aggregates, "refresh_summary_snapshots", summary)

    with pytest.raises(RuntimeError, match="maintenance incomplete"):
        await dashboard_aggregates.reconcile_dashboard_aggregates(
            pool, as_of=datetime(2026, 9, 16, 12, tzinfo=UTC)
        )
    summary.assert_awaited_once()


@pytest.mark.asyncio
async def test_backlog_status_failure_fails_maintenance_after_hourly_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = MagicMock()
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall.return_value = []
    conn.execute.return_value = cursor
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = conn
    pool.connection.return_value = connection_context
    hour = datetime(2026, 9, 16, 10, tzinfo=UTC)
    monkeypatch.setattr(
        dashboard_aggregates, "pending_hourly_aggregates", AsyncMock(return_value=[hour])
    )
    monkeypatch.setattr(
        dashboard_aggregates,
        "prepare_hourly_aggregate_catalog",
        AsyncMock(return_value={}),
    )
    writer = AsyncMock()
    monkeypatch.setattr(dashboard_aggregates, "_refresh_hourly_aggregate", writer)
    status = AsyncMock(side_effect=RuntimeError("status query failed"))
    monkeypatch.setattr(dashboard_aggregates, "pending_hourly_status", status)
    summary = AsyncMock(return_value=RefreshResult("published", 2))
    monkeypatch.setattr(dashboard_aggregates, "refresh_summary_snapshots", summary)

    with pytest.raises(RuntimeError, match="maintenance incomplete"):
        await dashboard_aggregates.reconcile_dashboard_aggregates(
            pool, as_of=datetime(2026, 9, 16, 12, tzinfo=UTC)
        )
    writer.assert_awaited_once()
    summary.assert_awaited_once()
    status.assert_awaited_once()


@pytest.mark.asyncio
async def test_owned_deadline_keeps_unselected_backlog_in_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExpiringTimeout:
        async def __aenter__(self) -> "ExpiringTimeout":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        def expired(self) -> bool:
            return True

    pool = MagicMock()
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall.return_value = []
    conn.execute.return_value = cursor
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = conn
    pool.connection.return_value = connection_context
    first = datetime(2026, 9, 16, 10, tzinfo=UTC)
    second = first + timedelta(hours=1)
    monkeypatch.setattr(asyncio, "timeout", lambda _seconds: ExpiringTimeout())
    monkeypatch.setattr(
        dashboard_aggregates,
        "pending_hourly_aggregates",
        AsyncMock(return_value=[first, second]),
    )
    status = AsyncMock(return_value=PendingHourlyStatus(4, second))
    monkeypatch.setattr(dashboard_aggregates, "pending_hourly_status", status)
    monkeypatch.setattr(
        dashboard_aggregates,
        "prepare_hourly_aggregate_catalog",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        dashboard_aggregates,
        "_refresh_hourly_aggregate",
        AsyncMock(side_effect=[None, TimeoutError("phase expired")]),
    )
    monkeypatch.setattr(
        dashboard_aggregates,
        "refresh_summary_snapshots",
        AsyncMock(return_value=RefreshResult("published", 5)),
    )

    result = await dashboard_aggregates.reconcile_dashboard_aggregates(
        pool, as_of=datetime(2026, 9, 16, 12, tzinfo=UTC)
    )

    assert result.committed == 1
    assert result.remaining == 4
    assert result.stop_reason == "deadline"
    status.assert_awaited_once()


@pytest.mark.asyncio
async def test_source_failure_does_not_emit_hourly_failure_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = MagicMock()
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall.return_value = [{"bucket_at": datetime(2026, 9, 16, 10, tzinfo=UTC)}]
    conn.execute.return_value = cursor
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = conn
    pool.connection.return_value = connection_context
    monkeypatch.setattr(
        dashboard_aggregates, "rebuild_source_bucket", AsyncMock(side_effect=RuntimeError("source"))
    )
    monkeypatch.setattr(
        dashboard_aggregates, "pending_hourly_aggregates", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        dashboard_aggregates,
        "refresh_summary_snapshots",
        AsyncMock(return_value=RefreshResult("published", 1)),
    )

    with capture_logs() as logs, pytest.raises(RuntimeError, match="maintenance incomplete"):
        await dashboard_aggregates.reconcile_dashboard_aggregates(
            pool, as_of=datetime(2026, 9, 16, 12, tzinfo=UTC)
        )
    assert not any(log["event"] == "dashboard_hourly_reconciliation_failed" for log in logs)


@pytest.mark.asyncio
async def test_summary_failure_does_not_emit_hourly_failure_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = MagicMock()
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall.return_value = []
    conn.execute.return_value = cursor
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = conn
    pool.connection.return_value = connection_context
    monkeypatch.setattr(
        dashboard_aggregates, "pending_hourly_aggregates", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        dashboard_aggregates,
        "refresh_summary_snapshots",
        AsyncMock(side_effect=RuntimeError("summary")),
    )

    with capture_logs() as logs, pytest.raises(RuntimeError, match="maintenance incomplete"):
        await dashboard_aggregates.reconcile_dashboard_aggregates(
            pool, as_of=datetime(2026, 9, 16, 12, tzinfo=UTC)
        )
    assert not any(log["event"] == "dashboard_hourly_reconciliation_failed" for log in logs)


@pytest.mark.asyncio
async def test_database_maintenance_commits_first_hour_when_second_fails(
    pg_conn: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    apply_migrations(pg_conn)
    pg_conn.autocommit = True
    first = datetime(2026, 9, 16, 10, tzinfo=UTC)
    second = first + timedelta(hours=1)
    _bucket(pg_conn, first, [("TTFT", "primary", "seconds", 10.0, 1)])
    _bucket(pg_conn, second, [("TTFT", "primary", "seconds", 20.0, 1)])
    with pg_conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO benchmarks_v2.dashboard_hourly_state
            (hour_at, dirty, refreshed_at, definition_revision, definition_fingerprint)
            VALUES (%s, true, NULL, %s, %s)""",
            [
                (first, DEFINITION_REVISION, aggregation_fingerprint()),
                (second, DEFINITION_REVISION, aggregation_fingerprint()),
            ],
        )
    pool = await storage._pool(pg_conn)
    summary = AsyncMock(return_value=RefreshResult("published", 9))
    real_writer = dashboard_hourly._refresh_hourly_aggregate
    monkeypatch.setattr(
        dashboard_aggregates,
        "pending_hourly_aggregates",
        AsyncMock(return_value=[first, second]),
    )

    async def fail_second(pool_arg: Any, hour: datetime, **kwargs: Any) -> None:
        if floor_hour(hour) == second:
            raise RuntimeError("second hour failed")
        await real_writer(pool_arg, hour, **kwargs)

    monkeypatch.setattr(dashboard_aggregates, "_refresh_hourly_aggregate", fail_second)
    monkeypatch.setattr(dashboard_aggregates, "refresh_summary_snapshots", summary)
    try:
        with pytest.raises(RuntimeError, match="maintenance incomplete"):
            await dashboard_aggregates.reconcile_dashboard_aggregates(
                pool, as_of=second + timedelta(hours=1)
            )
        async with pool.connection() as conn:
            states = await (
                await conn.execute(
                    "SELECT hour_at, dirty FROM benchmarks_v2.dashboard_hourly_state "
                    "WHERE hour_at IN (%s, %s) ORDER BY hour_at",
                    (first, second),
                )
            ).fetchall()
        assert states == [{"hour_at": first, "dirty": False}, {"hour_at": second, "dirty": True}]
        summary.assert_awaited_once()
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_external_cancellation_keeps_hour_dirty_and_skips_summary(
    pg_conn: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    apply_migrations(pg_conn)
    pg_conn.autocommit = True
    hour = datetime(2026, 9, 16, 10, tzinfo=UTC)
    _bucket(pg_conn, hour, [("TTFT", "primary", "seconds", 10.0, 1)])
    pg_conn.execute(
        """INSERT INTO benchmarks_v2.dashboard_hourly_state
        (hour_at, dirty, refreshed_at, definition_revision, definition_fingerprint)
        VALUES (%s, true, NULL, %s, %s)""",
        (hour, DEFINITION_REVISION, aggregation_fingerprint()),
    )
    pool = await storage._pool(pg_conn)
    summary = AsyncMock(return_value=RefreshResult("published", 1))
    transaction_updated = asyncio.Event()
    never_finish = asyncio.Event()
    real_refresh = dashboard_hourly._refresh_hours_in_transaction

    async def pause_before_commit(cur: Any, hours: list[datetime], **kwargs: Any) -> None:
        await real_refresh(cur, hours, **kwargs)
        transaction_updated.set()
        await never_finish.wait()

    monkeypatch.setattr(
        dashboard_aggregates,
        "pending_hourly_aggregates",
        AsyncMock(return_value=[hour]),
    )
    monkeypatch.setattr(dashboard_hourly, "_refresh_hours_in_transaction", pause_before_commit)
    monkeypatch.setattr(dashboard_aggregates, "refresh_summary_snapshots", summary)
    try:
        maintenance = asyncio.create_task(
            dashboard_aggregates.reconcile_dashboard_aggregates(
                pool, as_of=hour + timedelta(hours=1)
            )
        )
        await asyncio.wait_for(transaction_updated.wait(), timeout=5)
        maintenance.cancel()
        with pytest.raises(asyncio.CancelledError):
            await maintenance
        async with pool.connection() as conn:
            state = await (
                await conn.execute(
                    "SELECT dirty FROM benchmarks_v2.dashboard_hourly_state WHERE hour_at=%s",
                    (hour,),
                )
            ).fetchone()
            aggregates = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM benchmarks_v2.dashboard_hourly_aggregates "
                    "WHERE hour_at=%s",
                    (hour,),
                )
            ).fetchone()
        assert state == {"dirty": True}
        assert aggregates == {"n": 0}
        summary.assert_not_awaited()
    finally:
        await pool.close()


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
@pytest.mark.parametrize("status", [RunStatus.SUCCEEDED, RunStatus.PARTIAL, RunStatus.FAILED])
async def test_finish_enqueues_source_then_rebuild_marks_hour_dirty(
    pg_conn: psycopg.Connection[Any],
    status: RunStatus,
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
        await writer.finish_run(run_id, status=status)
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
        assert run["status"] == str(status) and run["finished_at"] is not None
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
