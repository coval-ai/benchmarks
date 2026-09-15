# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Metric identities retain references without changing raw metric codes."""

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db.dashboard_hourly import refresh_hourly_aggregates
from coval_bench.db.metric_definitions import (
    register_metric_definitions,
    register_metric_definitions_sync,
)
from coval_bench.registries.metrics import (
    METRIC_SPECS,
    METRIC_VALUE_CONTRACTS,
    TIMELINE_AGGREGATION_RULES,
    Metric,
)

from .conftest import apply_migrations, open_pool
from .test_dashboard_hourly import _bucket

metric_pg = postgresql("pg_proc")

_INSERT_HOURLY = """
INSERT INTO benchmarks_v2.dashboard_hourly_aggregates
(provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant,
 hour_at, primary_sum, sample_count, coverage_complete, source_count, definition_revision)
VALUES ('p', 'm', 'STT', 'd', %s, %s, %s, '2026-09-14 12:00:00+00', 10, 1, true, 1, 2)
"""


def test_frozen_seeds_have_generated_ids_and_current_display_names(
    metric_pg: psycopg.Connection[Any],
) -> None:
    apply_migrations(metric_pg)
    rows = metric_pg.execute("SELECT id, code, display_name FROM benchmarks_v2.metrics").fetchall()
    assert len(rows) == 12
    assert len({row[0] for row in rows}) == 12
    assert all(isinstance(row[0], int) and row[0] > 0 for row in rows)
    assert {row[1]: row[2] for row in rows} == {
        metric.value: spec.display_name for metric, spec in METRIC_SPECS.items()
    }
    identity = metric_pg.execute(
        """SELECT data_type, identity_generation FROM information_schema.columns
           WHERE table_schema='benchmarks_v2' AND table_name='metrics' AND column_name='id'"""
    ).fetchone()
    assert identity == ("bigint", "ALWAYS")
    for table in (
        "dashboard_hourly_aggregates",
        "normalized_results_24h",
        "normalized_results_7d",
        "normalized_results_30d",
    ):
        columns = metric_pg.execute(
            """SELECT attname FROM pg_attribute
               WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped""",
            (f"benchmarks_v2.{table}",),
        ).fetchall()
        assert ("metric_id",) in columns and ("metric_type",) in columns


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE benchmarks_v2.metrics SET code='Renamed' WHERE code='WER'",
        "UPDATE benchmarks_v2.metrics SET id=DEFAULT WHERE code='WER'",
        "DELETE FROM benchmarks_v2.metrics WHERE code='WER'",
        "TRUNCATE benchmarks_v2.metrics CASCADE",
    ],
)
def test_identity_and_retention_are_enforced(
    metric_pg: psycopg.Connection[Any], statement: str
) -> None:
    apply_migrations(metric_pg)
    metric_pg.autocommit = True
    before = metric_pg.execute("SELECT id, code FROM benchmarks_v2.metrics ORDER BY id").fetchall()
    with pytest.raises(psycopg.errors.CheckViolation), metric_pg.transaction():
        metric_pg.execute(statement)
    assert (
        metric_pg.execute("SELECT id, code FROM benchmarks_v2.metrics ORDER BY id").fetchall()
        == before
    )


def test_hourly_foreign_key_and_version_variant_identity(
    metric_pg: psycopg.Connection[Any],
) -> None:
    apply_migrations(metric_pg)
    metric_pg.autocommit = True
    row = metric_pg.execute("SELECT id FROM benchmarks_v2.metrics WHERE code='WER'").fetchone()
    assert row is not None
    metric_id = row[0]
    for version, variant in (("v1", "default"), ("v2", "default"), ("v1", "candidate")):
        metric_pg.execute(_INSERT_HOURLY, (metric_id, version, variant))
    assert metric_pg.execute(
        "SELECT count(*) FROM benchmarks_v2.dashboard_hourly_aggregates"
    ).fetchone() == (3,)
    with pytest.raises(psycopg.errors.ForeignKeyViolation), metric_pg.transaction():
        metric_pg.execute(_INSERT_HOURLY, (-1, "v1", "default"))
    with pytest.raises(psycopg.errors.UniqueViolation), metric_pg.transaction():
        metric_pg.execute(_INSERT_HOURLY, (metric_id, "v1", "default"))
    with pytest.raises(psycopg.errors.ForeignKeyViolation), metric_pg.transaction():
        metric_pg.execute("SELECT benchmarks_v2.metric_id_for_code('UnregisteredMetric')")


@pytest.mark.asyncio
async def test_registration_retains_ids_and_deliberate_display_changes(
    metric_pg: psycopg.Connection[Any],
) -> None:
    apply_migrations(metric_pg)
    metric_pg.autocommit = True
    pool = await open_pool(metric_pg)
    try:
        async with pool.connection() as conn, conn.transaction():
            before = await register_metric_definitions(conn)
            await conn.execute(
                "UPDATE benchmarks_v2.metrics SET display_name='Error rate' WHERE code='WER'"
            )
            after = await register_metric_definitions(conn)
            assert after == before
            label = await (
                await conn.execute(
                    "SELECT display_name FROM benchmarks_v2.metrics WHERE code='WER'"
                )
            ).fetchone()
            assert label == {"display_name": "Error rate"}
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_future_metric_is_registered_before_hourly_publication(
    metric_pg: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    class FutureMetric(StrEnum):
        LATENCY = "FutureLatency"

    apply_migrations(metric_pg)
    metric_pg.autocommit = True
    future = cast(Metric, FutureMetric.LATENCY)
    spec = METRIC_SPECS[Metric.TTFT].model_copy(update={"display_name": "Future latency"})
    contract = METRIC_VALUE_CONTRACTS[(Metric.TTFT, "v1")].model_copy(update={"metric": future})
    monkeypatch.setitem(METRIC_SPECS, future, spec)
    monkeypatch.setitem(METRIC_VALUE_CONTRACTS, (future, "v1"), contract)
    monkeypatch.setitem(TIMELINE_AGGREGATION_RULES, future.value, contract)
    hour = datetime(2026, 9, 14, 12, tzinfo=UTC)
    assert (
        metric_pg.execute(
            "SELECT id FROM benchmarks_v2.metrics WHERE code=%s", (future.value,)
        ).fetchone()
        is None
    )
    registered_id = register_metric_definitions_sync(metric_pg)[future.value]
    _bucket(metric_pg, hour, [(future.value, "primary", "seconds", 12.0, 3)])
    pool = await open_pool(metric_pg)
    try:
        assert await refresh_hourly_aggregates(pool, hours=[hour]) == 1
        first = metric_pg.execute(
            "SELECT id FROM benchmarks_v2.metrics WHERE code=%s", (future.value,)
        ).fetchone()
        assert first is not None and first[0] == registered_id
        assert await refresh_hourly_aggregates(pool, hours=[hour]) == 1
        assert (
            metric_pg.execute(
                "SELECT id FROM benchmarks_v2.metrics WHERE code=%s", (future.value,)
            ).fetchone()
            == first
        )
        rows = metric_pg.execute(
            """SELECT m.code, a.metric_id, a.primary_sum, a.sample_count
               FROM benchmarks_v2.dashboard_hourly_aggregates a
               JOIN benchmarks_v2.metrics m ON m.id=a.metric_id"""
        ).fetchall()
        assert rows == [(future.value, first[0], 12.0, 3)]
        assert metric_pg.execute(
            "SELECT DISTINCT metric_type FROM benchmarks_v2.metric_values_by_bucket"
        ).fetchall() == [(future.value,)]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_concurrent_registration_uses_one_catalog_identity(
    metric_pg: psycopg.Connection[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FutureMetric(StrEnum):
        VALUE = "ConcurrentMetric"

    apply_migrations(metric_pg)
    metric_pg.autocommit = True
    monkeypatch.setitem(
        METRIC_SPECS,
        cast(Metric, FutureMetric.VALUE),
        METRIC_SPECS[Metric.TTFT].model_copy(update={"display_name": "Concurrent metric"}),
    )
    pool = await open_pool(metric_pg)
    barrier = asyncio.Barrier(2)

    async def allocate() -> int:
        async with pool.connection() as conn, conn.transaction():
            await barrier.wait()
            return (await register_metric_definitions(conn))[FutureMetric.VALUE]

    try:
        first, second = await asyncio.wait_for(asyncio.gather(allocate(), allocate()), timeout=5)
    finally:
        await pool.close()
    assert first == second
    assert metric_pg.execute(
        "SELECT id,display_name FROM benchmarks_v2.metrics WHERE code='ConcurrentMetric'"
    ).fetchall() == [(first, "Concurrent metric")]
