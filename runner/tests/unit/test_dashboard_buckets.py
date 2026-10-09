"""Closed-bucket rollup fills: closure, grouping, idempotency, and the fill ledger."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db.dashboard_buckets import (
    BUCKET_INTERVALS,
    fill_bucket,
    fill_buckets_covering,
    fill_closed_buckets,
    floor_bucket,
    is_closed,
    missing_buckets,
)
from coval_bench.db.models import RunStatus
from coval_bench.db.writer import RunWriter
from tests.unit import test_normalized_db_writer as storage
from tests.unit.conftest import apply_migrations

pg_conn = postgresql("pg_proc")


def _bucket(
    conn: psycopg.Connection[Any], hour: datetime, values: list[tuple[str, str, str, float, int]]
) -> None:
    """Insert compact normalized per-run source rows for one slot."""
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO benchmarks_v2.metric_values_by_bucket
        (provider, model, benchmark, dataset_id, metric_type, metric_version,
         evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50,
         p75, max_value, value_sum, sample_count)
            VALUES ('p','m','STT','d',%s,'v1','default',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [
                (metric, key, unit, hour, val, val, val, val, val, val, count)
                for metric, key, unit, val, count in values
            ],
        )


def test_floor_and_closure_follow_the_interval() -> None:
    at = datetime(2026, 9, 14, 13, 59, 41, tzinfo=UTC)
    assert floor_bucket(at, 3600) == datetime(2026, 9, 14, 13, tzinfo=UTC)
    assert floor_bucket(at, 14400) == datetime(2026, 9, 14, 12, tzinfo=UTC)
    assert floor_bucket(at.replace(tzinfo=None), 3600) == datetime(2026, 9, 14, 13, tzinfo=UTC)
    hour = datetime(2026, 9, 14, 13, tzinfo=UTC)
    assert not is_closed(hour, 3600, as_of=hour + timedelta(hours=1, minutes=4))
    assert is_closed(hour, 3600, as_of=hour + timedelta(hours=1, minutes=5))


@pytest.mark.asyncio
async def test_fill_groups_observations_per_dataset_and_pooled(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        # Two runs share the slot; both land in the same bucket once finished.
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

        await fill_bucket(pool, interval_seconds=3600, bucket_at=storage._NOW)
        async with pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """SELECT dataset_id, value_key, value_sum, sample_count, p50, p95,
                              max_value, latest_source_at
                       FROM benchmarks_v2.dashboard_bucket_aggregates
                       WHERE interval_seconds = 3600 AND bucket_at = %s
                       ORDER BY dataset_id, value_key""",
                    (storage._NOW,),
                )
            ).fetchall()
            fills = await (
                await conn.execute("SELECT * FROM benchmarks_v2.dashboard_bucket_fills")
            ).fetchall()
        assert [(r["dataset_id"], r["value_key"]) for r in rows] == [
            ("__all__", "deletions"),
            ("__all__", "insertions"),
            ("__all__", "primary"),
            ("__all__", "substitutions"),
            ("observation-dataset", "deletions"),
            ("observation-dataset", "insertions"),
            ("observation-dataset", "primary"),
            ("observation-dataset", "substitutions"),
        ]
        primary = rows[2]
        assert (primary["value_sum"], primary["sample_count"]) == (40.0, 2)
        assert (primary["p50"], primary["p95"], primary["max_value"]) == (20.0, 29.0, 30.0)
        assert primary["latest_source_at"] == storage._NOW
        assert [(f["interval_seconds"], f["bucket_at"]) for f in fills] == [(3600, storage._NOW)]

        # Refilling replaces rather than duplicates, and the ledger row survives.
        await fill_bucket(pool, interval_seconds=3600, bucket_at=storage._NOW)
        async with pool.connection() as conn:
            count = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM benchmarks_v2.dashboard_bucket_aggregates"
                )
            ).fetchone()
        assert count == {"n": 8}
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
        pending = await missing_buckets(pool, interval_seconds=3600, as_of=as_of)
        assert pending[-1] == hour  # hour + 1 is not closed yet
        assert pending[0] == floor_bucket(as_of - timedelta(days=30), 3600)

        await fill_bucket(pool, interval_seconds=3600, bucket_at=hour)
        assert hour not in await missing_buckets(pool, interval_seconds=3600, as_of=as_of)

        writer = RunWriter(pool)
        await writer.start_run(
            dataset_id="d", dataset_sha256=storage._SHA, scheduled_at=hour - timedelta(minutes=30)
        )
        pending = await missing_buckets(pool, interval_seconds=3600, as_of=as_of)
        assert hour - timedelta(hours=1) not in pending
        assert hour - timedelta(hours=2) in pending
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
        expired = await fill_closed_buckets(
            pool, as_of=as_of, deadline=asyncio.get_running_loop().time() - 1
        )
        assert expired.filled == 0 and expired.remaining > 0
        full = await fill_closed_buckets(pool, as_of=as_of)
        assert full.remaining == 0 and full.filled == expired.remaining
        again = await fill_closed_buckets(pool, as_of=as_of)
        assert again == type(again)(0, 0)
        async with pool.connection() as conn:
            fills = await (
                await conn.execute(
                    "SELECT interval_seconds, count(*) AS n"
                    " FROM benchmarks_v2.dashboard_bucket_fills"
                    " GROUP BY interval_seconds ORDER BY interval_seconds"
                )
            ).fetchall()
        assert [f["interval_seconds"] for f in fills] == list(BUCKET_INTERVALS)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_fill_buckets_covering_only_touches_closed_buckets(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        slot = storage._NOW + timedelta(minutes=30)
        assert await fill_buckets_covering(pool, [slot], as_of=slot) == 0
        assert await fill_buckets_covering(pool, [slot], as_of=slot + timedelta(hours=1)) == 1
        assert await fill_buckets_covering(pool, [slot], as_of=slot + timedelta(hours=4)) == 2
    finally:
        await pool.close()
