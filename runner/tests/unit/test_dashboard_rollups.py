"""Rollup rebuilds: grouping, the slot queue, draining, retries, and retention."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db import dashboard_rollups
from coval_bench.db.dashboard_rollups import (
    DrainResult,
    drain_rollup_queue,
    floor_rollup,
)
from coval_bench.db.models import RunStatus
from coval_bench.db.writer import RunWriter
from tests.unit import test_normalized_db_writer as storage
from tests.unit.conftest import apply_migrations

pg_conn = postgresql("pg_proc")


def test_floor_follows_the_grain() -> None:
    at = datetime(2026, 9, 14, 13, 59, 41, tzinfo=UTC)
    assert floor_rollup(at, "1h") == datetime(2026, 9, 14, 13, tzinfo=UTC)
    assert floor_rollup(at, "4h") == datetime(2026, 9, 14, 12, tzinfo=UTC)
    assert floor_rollup(at, "run") == at
    assert floor_rollup(at.replace(tzinfo=None), "1h") == datetime(2026, 9, 14, 13, tzinfo=UTC)


async def _seed_two_runs(pool: Any) -> int:
    writer = RunWriter(pool)
    run_id, first = await storage._observation(writer, sample="a")
    second_run_id, second = await storage._observation(writer, sample="b")
    for observation, value in ((first, 10.0), (second, 30.0)):
        evaluation = await storage._evaluation(writer, observation)
        evaluation_id = storage._required(evaluation.id)
        scale = value / 10.0
        values = [
            item.model_copy(update={"value": item.value * scale})
            for item in storage._wer_values(evaluation_id)
        ]
        await writer.complete_metric_evaluation(
            evaluation_id, values=values, finished_at=storage._NOW
        )
    await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
    await writer.finish_run(second_run_id, status=RunStatus.PARTIAL)
    return run_id


async def _rows(pool: Any, sql: str, *params: Any) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        return list(await (await conn.execute(sql, params)).fetchall())


@pytest.mark.asyncio
async def test_finished_run_rebuilds_its_slot_and_queues_the_buckets(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        run_id = await _seed_two_runs(pool)
        await RunWriter(pool).rebuild_run_rollup(run_id)

        rows = await _rows(
            pool,
            """SELECT dataset_id, value_key, value_sum, sample_count, p50, p95, max_value,
                      wer_error_words, latest_run_at
               FROM benchmarks_v2.dashboard_rollups WHERE grain = 'run'
               ORDER BY dataset_id, value_key""",
        )
        assert [(r["dataset_id"], r["value_key"]) for r in rows] == [
            ("__all__", "primary"),
            ("observation-dataset", "primary"),
        ]
        primary = rows[0]
        assert (primary["value_sum"], primary["sample_count"]) == (40.0, 2)
        assert (primary["p50"], primary["p95"], primary["max_value"]) == (20.0, 29.0, 30.0)
        assert primary["wer_error_words"] is None
        assert primary["latest_run_at"] == storage._NOW
        queued = await _rows(pool, "SELECT slot_at FROM benchmarks_v2.dashboard_rollup_queue")
        assert [q["slot_at"] for q in queued] == [storage._NOW]
        buckets = await _rows(
            pool, "SELECT count(*) AS n FROM benchmarks_v2.dashboard_rollups WHERE grain <> 'run'"
        )
        assert buckets == [{"n": 0}]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_drain_rebuilds_slot_and_both_buckets_then_dequeues(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        run_id = await _seed_two_runs(pool)
        await RunWriter(pool).rebuild_run_rollup(run_id)
        as_of = storage._NOW + timedelta(hours=5)

        assert await drain_rollup_queue(pool, as_of=as_of) == DrainResult(1, 0)
        grains = await _rows(
            pool,
            """SELECT grain, bucket_at, sample_count FROM benchmarks_v2.dashboard_rollups
               WHERE dataset_id = '__all__' ORDER BY grain""",
        )
        assert [(g["grain"], g["bucket_at"], g["sample_count"]) for g in grains] == [
            ("1h", storage._NOW, 2),
            ("4h", storage._NOW, 2),
            ("run", storage._NOW, 2),
        ]
        assert await _rows(pool, "SELECT 1 FROM benchmarks_v2.dashboard_rollup_queue") == []
        assert await drain_rollup_queue(pool, as_of=as_of) == DrainResult(0, 0)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_drain_stops_at_the_deadline_and_keeps_failed_slots_queued(
    pg_conn: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        slots = [storage._NOW + timedelta(hours=offset) for offset in (0, 1)]
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO benchmarks_v2.dashboard_rollup_queue (slot_at) VALUES (%s), (%s)",
                slots,
            )
        as_of = storage._NOW + timedelta(days=1)
        expired = await drain_rollup_queue(
            pool, as_of=as_of, deadline=asyncio.get_running_loop().time() - 1
        )
        assert expired == DrainResult(0, 2)

        monkeypatch.setattr(dashboard_rollups, "FILL_ROLLUP_SQL", "SELECT nope")
        with pytest.raises(psycopg.Error):
            await drain_rollup_queue(pool, as_of=as_of)
        queued = await _rows(pool, "SELECT slot_at FROM benchmarks_v2.dashboard_rollup_queue")
        assert [q["slot_at"] for q in queued] == slots
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_drain_prunes_each_grain_at_its_own_retention(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        run_id = await _seed_two_runs(pool)
        await RunWriter(pool).rebuild_run_rollup(run_id)
        eight_days_on = storage._NOW + timedelta(days=8)
        assert await drain_rollup_queue(pool, as_of=eight_days_on) == DrainResult(1, 0)
        grains = await _rows(
            pool, "SELECT DISTINCT grain FROM benchmarks_v2.dashboard_rollups ORDER BY grain"
        )
        assert [g["grain"] for g in grains] == ["4h", "run"]
        await drain_rollup_queue(pool, as_of=storage._NOW + timedelta(days=31))
        assert await _rows(pool, "SELECT 1 FROM benchmarks_v2.dashboard_rollups") == []
    finally:
        await pool.close()
