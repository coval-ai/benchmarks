"""Closed-bucket rollup fills: closure, grouping, idempotency, and the fill ledger."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db.dashboard_rollups import (
    GRAINS,
    RUN_SLOT,
    fill_closed_rollups,
    fill_rollup,
    floor_rollup,
    missing_rollups,
)
from coval_bench.db.models import RunStatus
from coval_bench.db.writer import RunWriter
from tests.unit import test_normalized_db_writer as storage
from tests.unit.conftest import apply_migrations

pg_conn = postgresql("pg_proc")


def test_floor_follows_the_interval() -> None:
    at = datetime(2026, 9, 14, 13, 59, 41, tzinfo=UTC)
    assert floor_rollup(at, "1h") == datetime(2026, 9, 14, 13, tzinfo=UTC)
    assert floor_rollup(at, "4h") == datetime(2026, 9, 14, 12, tzinfo=UTC)
    assert floor_rollup(at.replace(tzinfo=None), "1h") == datetime(2026, 9, 14, 13, tzinfo=UTC)


@pytest.mark.asyncio
async def test_fill_groups_observations_per_dataset_and_pooled(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
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

        await fill_rollup(pool, grain="1h", bucket_at=storage._NOW)
        async with pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """SELECT dataset_id, value_key, value_sum, sample_count, p50, p95,
                              max_value, wer_error_words, latest_run_at
                       FROM benchmarks_v2.dashboard_rollups
                       WHERE grain = '1h' AND bucket_at = %s
                       ORDER BY dataset_id, value_key""",
                    (storage._NOW,),
                )
            ).fetchall()
            fills = await (
                await conn.execute("SELECT * FROM benchmarks_v2.dashboard_rollup_fills")
            ).fetchall()
        assert [(r["dataset_id"], r["value_key"]) for r in rows] == [
            ("__all__", "primary"),
            ("observation-dataset", "primary"),
        ]
        primary = rows[0]
        assert (primary["value_sum"], primary["sample_count"]) == (40.0, 2)
        assert (primary["p50"], primary["p95"], primary["max_value"]) == (20.0, 29.0, 30.0)
        assert primary["latest_run_at"] == storage._NOW
        assert primary["wer_error_words"] is None  # no word counts were stored
        assert [(f["grain"], f["bucket_at"]) for f in fills] == [("1h", storage._NOW)]

        await fill_rollup(pool, grain="1h", bucket_at=storage._NOW)
        async with pool.connection() as conn:
            count = await (
                await conn.execute("SELECT count(*) AS n FROM benchmarks_v2.dashboard_rollups")
            ).fetchone()
        assert count == {"n": 2}

        await fill_rollup(pool, grain=RUN_SLOT, bucket_at=storage._NOW)
        async with pool.connection() as conn:
            slots = await (
                await conn.execute(
                    """SELECT dataset_id, p95
                       FROM benchmarks_v2.dashboard_rollups
                       WHERE grain = 'run' ORDER BY dataset_id"""
                )
            ).fetchall()
            ledger = await (
                await conn.execute("SELECT count(*) AS n FROM benchmarks_v2.dashboard_rollup_fills")
            ).fetchone()
        assert [(r["dataset_id"], r["p95"]) for r in slots] == [
            ("__all__", 29.0),
            ("observation-dataset", 29.0),
        ]
        assert ledger == {"n": 1}
        async with pool.connection() as conn:
            open_grains = await (
                await conn.execute("SELECT grain FROM benchmarks_v2.dashboard_rollup_fills")
            ).fetchall()
        assert [row["grain"] for row in open_grains] == ["run"]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_missing_buckets_skip_open_filled_and_in_progress_intervals(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        hour = storage._NOW
        as_of = hour + timedelta(hours=2)
        pending = await missing_rollups(pool, grain="1h", as_of=as_of)
        assert pending[-1] == hour  # hour + 1 is not closed yet
        assert pending[0] == floor_rollup(as_of - timedelta(days=30), "1h")

        await fill_rollup(pool, grain="1h", bucket_at=hour)
        assert hour not in await missing_rollups(pool, grain="1h", as_of=as_of)

        writer = RunWriter(pool)
        await writer.start_run(
            dataset_id="d", dataset_sha256=storage._SHA, scheduled_at=hour - timedelta(minutes=30)
        )
        pending = await missing_rollups(pool, grain="1h", as_of=as_of)
        assert hour - timedelta(hours=1) not in pending
        assert hour - timedelta(hours=2) in pending
        pending = await missing_rollups(pool, grain="1h", as_of=as_of + timedelta(hours=12))
        assert hour - timedelta(hours=1) in pending
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_fill_closed_buckets_commits_one_at_a_time_until_the_deadline(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        as_of = storage._NOW + timedelta(hours=2)
        expired = await fill_closed_rollups(
            pool, as_of=as_of, deadline=asyncio.get_running_loop().time() - 1
        )
        assert expired.filled == 0 and expired.remaining > 0
        full = await fill_closed_rollups(pool, as_of=as_of)
        assert full.remaining == 0 and full.filled == expired.remaining
        again = await fill_closed_rollups(pool, as_of=as_of)
        assert again == type(again)(0, 0)
        async with pool.connection() as conn:
            fills = await (
                await conn.execute(
                    "SELECT grain, count(*) AS n"
                    " FROM benchmarks_v2.dashboard_rollup_fills"
                    " GROUP BY grain ORDER BY grain"
                )
            ).fetchall()
        assert [f["grain"] for f in fills] == list(GRAINS)
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO benchmarks_v2.dashboard_rollup_fills (grain, bucket_at)"
                " VALUES ('1h', %s)",
                (as_of - timedelta(days=31),),
            )
        await fill_closed_rollups(pool, as_of=as_of)
        async with pool.connection() as conn:
            stale = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM benchmarks_v2.dashboard_rollup_fills"
                    " WHERE bucket_at <= %s",
                    (as_of - timedelta(days=31),),
                )
            ).fetchone()
        assert stale == {"n": 0}
    finally:
        await pool.close()
