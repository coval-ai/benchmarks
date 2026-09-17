# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone PostgreSQL checks for the v2 results capture-time index."""

from __future__ import annotations

from typing import Any, cast

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from tests.unit.test_normalized_metric_id_migration import _dsn, _migrate

pg_conn = postgresql("pg_proc")
_INDEX_NAME = "benchmark_observations_capture_order_idx"


@pytest.fixture
def seeded(pg_conn: Any) -> Any:
    _migrate(pg_conn, "20260916_0037")
    pg_conn.autocommit = True
    with pg_conn.transaction():
        run_id = pg_conn.execute(
            """
            INSERT INTO benchmarks_v2.runs
              (dataset_id, dataset_sha256, runner_sha, status, finished_at, scheduled_at)
            VALUES ('capture-index-test', %s, 'test', 'succeeded', now(), now())
            RETURNING id
            """,
            ("a" * 64,),
        ).fetchone()[0]
        pg_conn.execute(
            """
            INSERT INTO benchmarks_v2.benchmark_observations
              (run_id, dataset_id, dataset_sha256, sample_id, provider, model,
               benchmark, source_kind, status, captured_at)
            VALUES (%s, 'capture-index-test', %s, 'sample', 'provider', 'model',
                    'STT', 'dataset_audio', 'succeeded', now())
            """,
            (run_id, "b" * 64),
        )
    return pg_conn


def _index_names(conn: Any) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'benchmarks_v2'
            """
        ).fetchall()
    }


def _capture_index(conn: Any) -> tuple[bool, bool, int, str]:
    row = conn.execute(
        """
        SELECT i.indisvalid, i.indpred IS NULL, i.indnkeyatts,
               pg_get_indexdef(i.indexrelid)
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'benchmarks_v2' AND c.relname = %s
        """,
        (_INDEX_NAME,),
    ).fetchone()
    assert row is not None
    return cast(tuple[bool, bool, int, str], row)


def _assert_valid_capture_index(conn: Any) -> None:
    valid, full, key_count, definition = _capture_index(conn)
    assert valid is True
    assert full is True
    assert key_count == 1
    assert "(captured_at DESC)" in definition
    assert "id DESC" not in definition


def _sample_count(conn: Any) -> tuple[int]:
    row = conn.execute(
        "SELECT count(*) FROM benchmarks_v2.benchmark_observations WHERE sample_id = 'sample'"
    ).fetchone()
    assert row is not None
    return cast(tuple[int], row)


def test_capture_index_is_full_single_column_and_reversible(seeded: Any) -> None:
    before = _index_names(seeded)
    assert _sample_count(seeded) == (1,)

    _migrate(seeded, "20260917_0038")
    _assert_valid_capture_index(seeded)
    assert before <= _index_names(seeded)
    assert _sample_count(seeded) == (1,)

    _migrate(seeded, "20260916_0037", down=True)
    assert _INDEX_NAME not in _index_names(seeded)
    assert before <= _index_names(seeded)
    assert _sample_count(seeded) == (1,)


def test_capture_index_retries_after_cancelled_concurrent_build(seeded: Any) -> None:
    before = _index_names(seeded)
    blocker = psycopg.connect(_dsn(seeded), autocommit=False)
    builder: psycopg.Connection[Any] | None = None
    try:
        blocker.execute(
            "UPDATE benchmarks_v2.benchmark_observations "
            "SET sample_id = 'blocked-sample' WHERE sample_id = 'sample'"
        )
        builder = psycopg.connect(_dsn(seeded), autocommit=True)
        builder.execute("SET statement_timeout = '250ms'")
        with pytest.raises(psycopg.errors.QueryCanceled):
            builder.execute(
                "CREATE INDEX CONCURRENTLY "
                f"{_INDEX_NAME} ON benchmarks_v2.benchmark_observations (captured_at DESC)"
            )
        valid, full, key_count, _ = _capture_index(builder)
        assert valid is False
        assert full is True
        assert key_count == 1
        assert seeded.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "20260916_0037",
        )
    finally:
        if builder is not None:
            builder.close()
        blocker.rollback()
        blocker.close()

    _migrate(seeded, "20260917_0038")
    _assert_valid_capture_index(seeded)
    assert before <= _index_names(seeded)
    assert _sample_count(seeded) == (1,)


def test_capture_index_replaces_valid_same_name_index(seeded: Any) -> None:
    before = _index_names(seeded)
    seeded.execute(
        f"CREATE INDEX {_INDEX_NAME} ON benchmarks_v2.benchmark_observations (captured_at DESC)"
    )
    assert _INDEX_NAME in _index_names(seeded)

    _migrate(seeded, "20260917_0038")
    _assert_valid_capture_index(seeded)
    assert before <= _index_names(seeded)
    assert _sample_count(seeded) == (1,)
