# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S608
"""Exercise the metric-ID migration with the independently deployed application."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import psycopg.rows
import pytest
from alembic import command
from alembic.config import Config
from fastapi import HTTPException
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql
from sqlalchemy.exc import IntegrityError

from coval_bench.api.dashboard_snapshots import require_snapshot
from coval_bench.db.dashboard_summaries import refresh_summary_snapshots
from coval_bench.db.models import (
    Benchmark,
    Observation,
    ObservationSourceKind,
    ObservationStatus,
    RunStatus,
)
from coval_bench.db.writer import RunWriter

from . import test_normalized_db_writer as writer_seed
from .conftest import async_dsn, open_pool
from .test_dashboard_hourly import _bucket

metric_ids_pg = postgresql("pg_proc")
_HOUR = datetime(2026, 9, 14, 12, tzinfo=UTC)
_AS_OF = _HOUR + timedelta(hours=1)
_VIEWS = tuple(f"normalized_results_{window}" for window in ("24h", "7d", "30d"))
_RETAINED = (
    "dashboard_hourly_state",
    "dashboard_source_refreshes",
    "runs",
    "benchmark_observations",
    "metric_evaluations",
    "metric_values",
    "metric_values_by_bucket",
)


@pytest.fixture
def api_role(metric_ids_pg: psycopg.Connection[Any]) -> Iterator[None]:
    """Keep the cluster-wide role from leaking into other test databases."""
    metric_ids_pg.execute("CREATE ROLE api")
    metric_ids_pg.commit()
    try:
        yield
    finally:
        metric_ids_pg.rollback()
        metric_ids_pg.execute("DROP OWNED BY api")
        metric_ids_pg.execute("DROP ROLE api")
        metric_ids_pg.commit()


def _migrate(conn: psycopg.Connection[Any], revision: str, *, down: bool = False) -> None:
    # A read transaction on this fixture connection would block ALTER TABLE
    # through Alembic's separate connection.
    conn.commit()
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option(
        "sqlalchemy.url", async_dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    (command.downgrade if down else command.upgrade)(config, revision)


def _rows(conn: psycopg.Connection[Any], table: str) -> list[Any]:
    return [
        row[0]
        for row in conn.execute(
            f"SELECT to_jsonb(t) FROM benchmarks_v2.{table} t ORDER BY 1"
        ).fetchall()
    ]


def _code_rows(conn: psycopg.Connection[Any], table: str) -> list[Any]:
    return [
        row[0]
        for row in conn.execute(
            f"""SELECT to_jsonb(t) - 'metric_id' || jsonb_build_object('metric_type',m.code)
                FROM benchmarks_v2.{table} t
                JOIN benchmarks_v2.metrics m ON m.id=t.metric_id ORDER BY 1"""
        ).fetchall()
    ]


def _populated(conn: psycopg.Connection[Any]) -> list[bool]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT relispopulated FROM pg_class WHERE oid=ANY(%s::regclass[]) ORDER BY relname",
            ([f"benchmarks_v2.{view}" for view in _VIEWS],),
        ).fetchall()
    ]


def _seed_0034(conn: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    _migrate(conn, "20260914_0034")
    conn.autocommit = True
    monkeypatch.setattr(writer_seed, "_NOW", _HOUR)

    async def raw_writer() -> None:
        # RunWriter commits each operation itself; deferred evaluation-output
        # constraints require its normal, non-autocommit connection mode.
        pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
            async_dsn(conn),
            kwargs={"row_factory": psycopg.rows.dict_row},
            open=False,
            min_size=1,
            max_size=2,
        )
        await pool.open()
        try:
            writer = RunWriter(pool)
            run = await writer.start_run(
                dataset_id="d", dataset_sha256="a" * 64, scheduled_at=_HOUR
            )
            assert run.id is not None
            observation = await writer.insert_observation(
                Observation(
                    run_id=run.id,
                    dataset_id="d",
                    dataset_sha256="a" * 64,
                    sample_id="migration",
                    provider="p",
                    model="m",
                    benchmark=Benchmark.STT,
                    source_kind=ObservationSourceKind.DATASET_AUDIO,
                    captured_at=_HOUR,
                    status=ObservationStatus.SUCCEEDED,
                )
            )
            evaluation = await writer_seed._evaluation(writer, observation)
            assert evaluation.id is not None
            await writer.complete_metric_evaluation(
                evaluation.id,
                values=writer_seed._wer_values(evaluation.id),
                finished_at=_AS_OF,
            )
            await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)
        finally:
            await pool.close()

    asyncio.run(raw_writer())
    for version, variant, total in (
        ("v1", "default", 10),
        ("v2", "default", 20),
        ("v1", "candidate", 30),
    ):
        conn.execute(
            """INSERT INTO benchmarks_v2.dashboard_hourly_aggregates
            (provider,model,benchmark,dataset_id,metric_type,metric_version,evaluation_variant,
             hour_at,primary_sum,sample_count,numerator_sum,denominator_sum,coverage_complete,
             source_count,latest_source_at,definition_revision,metadata)
            VALUES ('p','m','STT','d','WER',%s,%s,%s,%s,2,2,10,true,1,%s,1,
                    '{"schema_version":1,"preserve":"hourly"}')""",
            (version, variant, _HOUR, total, _HOUR),
        )
    conn.execute(
        """INSERT INTO benchmarks_v2.dashboard_hourly_state
        (hour_at,dirty,refreshed_at,definition_revision,definition_fingerprint)
        VALUES (%s,false,%s,1,'old')""",
        (_HOUR, _HOUR),
    )
    _bucket(conn, _HOUR, [("WER", "primary", "percent", 10.0, 2)])
    conn.execute(
        """UPDATE benchmarks_v2.dashboard_summary_state SET generation=7,
        as_of=%s,published_at=%s,definition_fingerprint='old'""",
        (_AS_OF, _AS_OF),
    )
    for view in _VIEWS:
        conn.execute(f"REFRESH MATERIALIZED VIEW benchmarks_v2.{view}")
        assert _rows(conn, view), "The migration must start with populated summary data"
    assert all(_rows(conn, table) for table in _RETAINED)


def test_populated_upgrade_preserves_data_and_requires_republication(
    metric_ids_pg: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch, api_role: None
) -> None:
    conn = metric_ids_pg
    _seed_0034(conn, monkeypatch)
    hourly = _rows(conn, "dashboard_hourly_aggregates")
    retained = {table: _rows(conn, table) for table in _RETAINED}
    summaries = {view: _rows(conn, view) for view in _VIEWS}
    conn.execute("GRANT USAGE ON SCHEMA benchmarks_v2 TO api")
    _migrate(conn, "head")

    assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("20260914_0035",)
    assert conn.execute("SELECT count(*) FROM benchmarks_v2.metrics").fetchone() == (12,)
    assert _code_rows(conn, "dashboard_hourly_aggregates") == hourly
    assert {table: _rows(conn, table) for table in _RETAINED} == retained
    assert _populated(conn) == [False, False, False]
    assert conn.execute(
        """SELECT generation,as_of,published_at,definition_revision,definition_fingerprint
        FROM benchmarks_v2.dashboard_summary_state"""
    ).fetchone() == (7, None, None, 2, "uninitialized")

    async def publish() -> None:
        pool = await open_pool(conn)
        try:
            async with pool.connection() as reader:
                with pytest.raises(HTTPException) as failure:
                    await require_snapshot(reader)
                assert failure.value.status_code == 503
                assert failure.value.detail == "dashboard_snapshot_not_ready"
            result = await refresh_summary_snapshots(pool, as_of=_AS_OF)
            assert result.status == "published" and result.generation == 8
            async with pool.connection() as reader:
                snapshot = await require_snapshot(reader)
                assert snapshot.generation == 8 and snapshot.definition_revision == 1
        finally:
            await pool.close()

    asyncio.run(publish())
    assert _populated(conn) == [True, True, True]
    with conn.transaction():
        conn.execute("SET LOCAL ROLE api")
        for view in _VIEWS:
            rows = conn.execute(
                f"SELECT s.metric_type, m.code FROM benchmarks_v2.{view} s "
                "JOIN benchmarks_v2.metrics m ON m.id=s.metric_id"
            ).fetchall()
            assert rows and all(code == resolved for code, resolved in rows)
    assert {view: _code_rows(conn, view) for view in _VIEWS} == summaries
    assert _code_rows(conn, "dashboard_hourly_aggregates") == hourly
    assert {table: _rows(conn, table) for table in _RETAINED} == retained

    async def republish() -> None:
        pool = await open_pool(conn)
        try:
            result = await refresh_summary_snapshots(pool, as_of=_AS_OF)
            assert result.status == "published" and result.generation == 9
        finally:
            await pool.close()

    asyncio.run(republish())
    assert {view: _code_rows(conn, view) for view in _VIEWS} == summaries


def test_unknown_historical_code_rolls_back_schema_rows_views_and_revision(
    metric_ids_pg: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = metric_ids_pg
    _seed_0034(conn, monkeypatch)
    conn.execute("UPDATE benchmarks_v2.dashboard_hourly_aggregates SET metric_type='UnknownMetric'")
    tables = (*_RETAINED, "dashboard_hourly_aggregates", "dashboard_summary_state", *_VIEWS)
    before = {table: _rows(conn, table) for table in tables}
    with pytest.raises(IntegrityError, match="unknown metric definition"):
        _migrate(conn, "head")
    assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("20260914_0034",)
    assert conn.execute("SELECT to_regclass('benchmarks_v2.metrics')").fetchone() == (None,)
    assert conn.execute(
        """SELECT data_type FROM information_schema.columns WHERE table_schema='benchmarks_v2'
        AND table_name='dashboard_hourly_aggregates' AND column_name='metric_type'"""
    ).fetchone() == ("text",)
    assert _populated(conn) == [True, True, True]
    assert {table: _rows(conn, table) for table in tables} == before


def test_downgrade_retains_codes_and_values_and_allows_reupgrade(
    metric_ids_pg: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = metric_ids_pg
    _seed_0034(conn, monkeypatch)
    hourly = _rows(conn, "dashboard_hourly_aggregates")
    retained = {table: _rows(conn, table) for table in _RETAINED}
    _migrate(conn, "head")
    _migrate(conn, "20260914_0034", down=True)
    assert _rows(conn, "dashboard_hourly_aggregates") == hourly
    assert {table: _rows(conn, table) for table in _RETAINED} == retained
    assert conn.execute("SELECT to_regclass('benchmarks_v2.metrics')").fetchone() == (None,)
    assert conn.execute(
        """SELECT generation,as_of,published_at,definition_revision,definition_fingerprint
        FROM benchmarks_v2.dashboard_summary_state"""
    ).fetchone() == (7, None, None, 1, "uninitialized")
    assert _populated(conn) == [False, False, False]
    _migrate(conn, "head")
    assert _code_rows(conn, "dashboard_hourly_aggregates") == hourly
    assert {table: _rows(conn, table) for table in _RETAINED} == retained
    assert _populated(conn) == [False, False, False]


def _insert_hourly(
    conn: psycopg.Connection[Any],
    code: str | None,
    metric_id: int | None,
    *,
    version: str = "v1",
    variant: str = "default",
) -> None:
    conn.execute(
        """INSERT INTO benchmarks_v2.dashboard_hourly_aggregates
        (provider,model,benchmark,dataset_id,metric_type,metric_id,metric_version,
         evaluation_variant,hour_at,primary_sum,sample_count,coverage_complete,
         source_count,definition_revision)
        VALUES ('p','m','STT','d',%s,%s,%s,%s,%s,10,1,true,1,2)""",
        (code, metric_id, version, variant, _HOUR),
    )


def test_old_and_new_writes_share_identity_and_preserve_version_variant_keys(
    metric_ids_pg: psycopg.Connection[Any],
) -> None:
    conn = metric_ids_pg
    _migrate(conn, "head")
    conn.autocommit = True
    ids = dict(conn.execute("SELECT code,id FROM benchmarks_v2.metrics").fetchall())
    _insert_hourly(conn, "WER", None)
    _insert_hourly(conn, None, ids["WER"], version="v2")
    _insert_hourly(conn, "WER", ids["WER"], variant="candidate")
    assert conn.execute(
        "SELECT DISTINCT metric_type,metric_id FROM benchmarks_v2.dashboard_hourly_aggregates"
    ).fetchall() == [("WER", ids["WER"])]
    assert conn.execute(
        "SELECT metric_version,evaluation_variant FROM benchmarks_v2.dashboard_hourly_aggregates "
        "ORDER BY metric_version,evaluation_variant"
    ).fetchall() == [("v1", "candidate"), ("v1", "default"), ("v2", "default")]
    with pytest.raises(psycopg.errors.UniqueViolation), conn.transaction():
        _insert_hourly(conn, None, ids["WER"])

    # Updating either interface must update its counterpart, including rows
    # initially written by the other application version.
    conn.execute("UPDATE benchmarks_v2.dashboard_hourly_aggregates SET metric_type='TTFT'")
    assert conn.execute(
        "SELECT DISTINCT metric_id FROM benchmarks_v2.dashboard_hourly_aggregates"
    ).fetchall() == [(ids["TTFT"],)]
    conn.execute("UPDATE benchmarks_v2.dashboard_hourly_aggregates SET metric_id=%s", (ids["WER"],))
    assert conn.execute(
        "SELECT DISTINCT metric_type FROM benchmarks_v2.dashboard_hourly_aggregates"
    ).fetchall() == [("WER",)]

    # New generated definitions work through either interface without fixed IDs.
    conn.execute(
        "INSERT INTO benchmarks_v2.metrics(code,display_name) VALUES ('FutureMetric','Future')"
    )
    _insert_hourly(conn, "FutureMetric", None)
    hourly = _code_rows(conn, "dashboard_hourly_aggregates")
    _migrate(conn, "20260914_0034", down=True)
    assert _rows(conn, "dashboard_hourly_aggregates") == hourly
    _insert_hourly_0034 = """INSERT INTO benchmarks_v2.dashboard_hourly_aggregates
        SELECT provider,model,benchmark,dataset_id,'TTFT',metric_version,evaluation_variant,
        hour_at,primary_sum,sample_count,numerator_sum,denominator_sum,coverage_complete,
        source_count,latest_source_at,definition_revision,metadata
        FROM benchmarks_v2.dashboard_hourly_aggregates WHERE metric_type='FutureMetric'"""
    conn.execute(_insert_hourly_0034)
    assert conn.execute(
        "SELECT count(*) FROM benchmarks_v2.dashboard_hourly_aggregates WHERE metric_type='TTFT'"
    ).fetchone() == (1,)


@pytest.mark.parametrize(
    ("code", "metric", "error"),
    [
        ("UnknownMetric", None, psycopg.errors.ForeignKeyViolation),
        (None, "missing", psycopg.errors.ForeignKeyViolation),
        ("WER", "TTFT", psycopg.errors.CheckViolation),
        (None, None, psycopg.errors.NotNullViolation),
    ],
)
def test_invalid_identity_writes_roll_back(
    metric_ids_pg: psycopg.Connection[Any],
    code: str | None,
    metric: str | None,
    error: type[psycopg.Error],
) -> None:
    conn = metric_ids_pg
    _migrate(conn, "head")
    conn.autocommit = True
    ids = dict(conn.execute("SELECT code,id FROM benchmarks_v2.metrics").fetchall())
    metric_id = None if metric is None else ids.get(metric, -1)
    with pytest.raises(error), conn.transaction():
        _insert_hourly(conn, code, metric_id)
    assert _rows(conn, "dashboard_hourly_aggregates") == []


def test_conflicting_update_does_not_change_retained_identity(
    metric_ids_pg: psycopg.Connection[Any],
) -> None:
    conn = metric_ids_pg
    _migrate(conn, "head")
    conn.autocommit = True
    ids = dict(conn.execute("SELECT code,id FROM benchmarks_v2.metrics").fetchall())
    _insert_hourly(conn, "WER", None)
    before = _rows(conn, "dashboard_hourly_aggregates")
    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        conn.execute(
            "UPDATE benchmarks_v2.dashboard_hourly_aggregates SET metric_type='TTFT',metric_id=%s",
            (ids["TTFA"],),
        )
    assert _rows(conn, "dashboard_hourly_aggregates") == before
    with pytest.raises(psycopg.errors.ForeignKeyViolation), conn.transaction():
        conn.execute("UPDATE benchmarks_v2.dashboard_hourly_aggregates SET metric_id=-1")
    assert _rows(conn, "dashboard_hourly_aggregates") == before


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE benchmarks_v2.metrics SET code='Renamed' WHERE code='WER'",
        "UPDATE benchmarks_v2.metrics SET id=DEFAULT WHERE code='WER'",
        "DELETE FROM benchmarks_v2.metrics WHERE code='WER'",
        "TRUNCATE benchmarks_v2.metrics CASCADE",
    ],
)
def test_metric_definitions_retain_immutable_identity(
    metric_ids_pg: psycopg.Connection[Any], statement: str
) -> None:
    conn = metric_ids_pg
    _migrate(conn, "head")
    conn.autocommit = True
    before = _rows(conn, "metrics")
    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        conn.execute(statement)
    assert _rows(conn, "metrics") == before
