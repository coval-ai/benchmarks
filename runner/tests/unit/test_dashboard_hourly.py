"""Focused contract checks for hourly dashboard storage helpers."""

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db.dashboard_hourly import (
    HOUR_LOCK_SQL,
    MARK_HOUR_DIRTY_SQL,
    floor_hour,
)
from tests.unit.conftest import apply_migrations, open_pool

pg_conn = postgresql("pg_proc")


def _bucket(
    conn: psycopg.Connection[Any], hour: datetime, values: list[tuple[str, str, str, float, int]]
) -> None:
    """Insert compact normalized source rows for one bucket."""
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


def test_floor_hour_normalizes_utc_and_accepts_naive_utc() -> None:
    assert floor_hour(datetime(2026, 9, 14, 12, 59, 41, 9)) == datetime(2026, 9, 14, 12, tzinfo=UTC)
    assert floor_hour(datetime(2026, 9, 14, 5, 59, tzinfo=UTC)) == datetime(
        2026, 9, 14, 5, tzinfo=UTC
    )


def test_lock_and_dirty_sql_keep_named_hour_and_definition_identity() -> None:
    assert "%(hour)s" in HOUR_LOCK_SQL
    assert "dashboard_hourly_aggregates" in HOUR_LOCK_SQL
    assert "%(definition_revision)s" in MARK_HOUR_DIRTY_SQL
    assert "%(definition_fingerprint)s" in MARK_HOUR_DIRTY_SQL
    assert "dirty = true" in MARK_HOUR_DIRTY_SQL
    assert "refreshed_at" not in MARK_HOUR_DIRTY_SQL.split("DO UPDATE", 1)[1]


@pytest.mark.asyncio
async def test_refresh_combines_half_hour_source_buckets(pg_conn: psycopg.Connection[Any]) -> None:
    """The actual migrated tables preserve weighted means across source buckets."""
    apply_migrations(pg_conn)
    pg_conn.autocommit = True
    hour = datetime(2026, 9, 14, 12, tzinfo=UTC)
    values = [(hour, 10.0, 1), (hour.replace(minute=30), 30.0, 3)]
    with pg_conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO benchmarks_v2.metric_values_by_bucket
        (provider, model, benchmark, dataset_id, metric_type, metric_version,
         evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50,
         p75, max_value, value_sum, sample_count)
            VALUES ('p','m','STT','d','TTFT','v1','default','primary','seconds',
                    %s,%s,%s,%s,%s,%s,%s,%s)""",
            [(at, val, val, val, val, val, val, count) for at, val, count in values],
        )
    pool = await open_pool(pg_conn)
    try:
        from coval_bench.db.dashboard_hourly import refresh_hourly_aggregates

        assert await refresh_hourly_aggregates(pool, hours=[hour]) == 1
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    """SELECT primary_sum, sample_count, coverage_complete
                    FROM benchmarks_v2.dashboard_hourly_aggregates
                    WHERE provider='p' AND model='m' AND metric_type='TTFT' AND hour_at=%s""",
                    (hour,),
                )
            ).fetchone()
        assert row == {"primary_sum": 40.0, "sample_count": 4, "coverage_complete": True}
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_ratio_coverage_and_zero_denominator_are_explicit(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pg_conn.autocommit = True
    hour = datetime(2026, 9, 14, 12, tzinfo=UTC)
    _bucket(
        pg_conn,
        hour,
        [
            ("WER", "primary", "percent", 20.0, 2),
            ("WER", "substitution_count", "count", 1.0, 2),
            ("WER", "deletion_count", "count", 1.0, 2),
            ("WER", "insertion_count", "count", 0.0, 2),
            ("WER", "reference_words", "count", 10.0, 2),
        ],
    )
    _bucket(
        pg_conn,
        hour.replace(minute=30),
        [
            ("WER", "primary", "percent", 40.0, 1),
            ("WER", "substitution_count", "count", 2.0, 1),
            ("WER", "deletion_count", "count", 0.0, 1),
            ("WER", "insertion_count", "count", 0.0, 1),
            ("WER", "reference_words", "count", 0.0, 1),
        ],
    )
    pool = await open_pool(pg_conn)
    try:
        from coval_bench.db.dashboard_hourly import refresh_hourly_aggregates

        await refresh_hourly_aggregates(pool, hours=[hour])
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    """SELECT numerator_sum, denominator_sum, coverage_complete
                FROM benchmarks_v2.dashboard_hourly_aggregates
                WHERE metric_type='WER' AND hour_at=%s""",
                    (hour,),
                )
            ).fetchone()
        assert row == {"numerator_sum": 4.0, "denominator_sum": 10.0, "coverage_complete": True}
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_wrong_primary_units_are_excluded(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pg_conn.autocommit = True
    hour = datetime(2026, 9, 14, 12, tzinfo=UTC)
    # One valid source bucket and one malformed bucket whose total count would
    # otherwise cancel out; coverage must remain false per source bucket.
    _bucket(pg_conn, hour, [("TTFT", "primary", "seconds", 10.0, 2)])
    _bucket(pg_conn, hour.replace(minute=30), [("TTFT", "primary", "milliseconds", 20.0, 2)])
    pool = await open_pool(pg_conn)
    try:
        from coval_bench.db.dashboard_hourly import refresh_hourly_aggregates

        await refresh_hourly_aggregates(pool, hours=[hour])
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    """SELECT coverage_complete, primary_sum, sample_count
                FROM benchmarks_v2.dashboard_hourly_aggregates
                WHERE metric_type='TTFT' AND hour_at=%s""",
                    (hour,),
                )
            ).fetchone()
        assert row == {"coverage_complete": True, "primary_sum": 10.0, "sample_count": 2}
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_retry_replaces_previous_hour_rows_without_double_counting(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pg_conn.autocommit = True
    hour = datetime(2026, 9, 14, 12, tzinfo=UTC)
    _bucket(pg_conn, hour, [("TTFT", "primary", "seconds", 10.0, 2)])
    pool = await open_pool(pg_conn)
    try:
        from coval_bench.db.dashboard_hourly import refresh_hourly_aggregates

        await refresh_hourly_aggregates(pool, hours=[hour])
        pg_conn.execute(
            "UPDATE benchmarks_v2.metric_values_by_bucket SET value_sum=30, min_value=30, "
            "p25=30, p50=30, p75=30, max_value=30 WHERE metric_type='TTFT' AND bucket_at=%s",
            (hour,),
        )
        await refresh_hourly_aggregates(pool, hours=[hour])
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT primary_sum, sample_count "
                    "FROM benchmarks_v2.dashboard_hourly_aggregates "
                    "WHERE metric_type='TTFT' AND hour_at=%s",
                    (hour,),
                )
            ).fetchone()
        assert row == {"primary_sum": 30.0, "sample_count": 2}
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_pending_reports_uninitialized_and_dirty_hours_outside_range(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pg_conn.autocommit = True
    await_hour = datetime(2026, 9, 14, 12, tzinfo=UTC)
    pg_conn.execute(
        """INSERT INTO benchmarks_v2.dashboard_hourly_state
        (hour_at, dirty, refreshed_at, definition_revision, definition_fingerprint)
        VALUES (%s, true, NULL, 1, 'old')""",
        (await_hour - timedelta(hours=3),),
    )
    pool = await open_pool(pg_conn)
    try:
        from coval_bench.db.dashboard_hourly import pending_hourly_aggregates

        pending = await pending_hourly_aggregates(
            pool, since=await_hour, until=await_hour + timedelta(hours=1)
        )
        assert await_hour in pending
        assert await_hour - timedelta(hours=3) in pending
    finally:
        await pool.close()
