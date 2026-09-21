# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone PostgreSQL checks for the normalized importer dedup index."""

from __future__ import annotations

from typing import Any, cast

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from tests.unit.test_normalized_metric_id_migration import _dsn, _migrate

pg_conn = postgresql("pg_proc")
_INDEX_NAME = "benchmark_observations_coval_ingest_idx"


@pytest.fixture
def seeded(pg_conn: Any) -> Any:
    _migrate(pg_conn, "20260918_0040")
    pg_conn.autocommit = True
    with pg_conn.transaction():
        run_id = pg_conn.execute(
            """
            INSERT INTO benchmarks_v2.runs
              (dataset_id, dataset_sha256, runner_sha, status, finished_at, scheduled_at)
            VALUES ('dedup-index-test', %s, 'test', 'succeeded', now(), now())
            RETURNING id
            """,
            ("a" * 64,),
        ).fetchone()[0]
        pg_conn.execute(
            """
            INSERT INTO benchmarks_v2.benchmark_observations
              (run_id, dataset_id, dataset_sha256, sample_id, provider, model,
               benchmark, source_kind, status)
            VALUES (%s, 'dedup-index-test', %s, 'external/simulation', 'provider', 'model',
                    'S2S', 'conversation_audio', 'succeeded')
            """,
            (run_id, "b" * 64),
        )
    return pg_conn


def _index_names(conn: Any) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'benchmarks_v2'"
        ).fetchall()
    }


def _index_info(conn: Any) -> tuple[bool, bool, bool, int, str, str]:
    row = conn.execute(
        """
        SELECT i.indisvalid, i.indisready, i.indpred IS NULL, i.indnkeyatts,
               pg_get_indexdef(i.indexrelid), pg_get_expr(i.indexprs, i.indrelid)
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'benchmarks_v2' AND c.relname = %s
        """,
        (_INDEX_NAME,),
    ).fetchone()
    assert row is not None
    return cast(tuple[bool, bool, bool, int, str, str], row)


def _assert_valid_index(conn: Any) -> None:
    valid, ready, full, key_count, definition, expressions = _index_info(conn)
    canonical_definition = definition.replace("::text", "")
    assert valid is True
    assert ready is True
    assert full is True
    assert key_count == 3
    assert "provider, benchmark, split_part(sample_id, '/', 1)" in canonical_definition
    assert "INCLUDE (id, run_id)" in canonical_definition
    assert "WHERE" not in canonical_definition.upper()
    assert expressions is not None
    assert expressions.replace("::text", "") == "split_part(sample_id, '/', 1)"


def _observation_count(conn: Any) -> tuple[int]:
    row = conn.execute(
        "SELECT count(*) FROM benchmarks_v2.benchmark_observations "
        "WHERE sample_id = 'external/simulation'"
    ).fetchone()
    assert row is not None
    return cast(tuple[int], row)


def test_coval_ingest_index_is_full_covering_and_reversible(seeded: Any) -> None:
    before = _index_names(seeded)
    assert _observation_count(seeded) == (1,)

    _migrate(seeded, "20260921_0041")
    _assert_valid_index(seeded)
    assert before <= _index_names(seeded)
    assert _observation_count(seeded) == (1,)

    _migrate(seeded, "20260918_0040", down=True)
    assert _INDEX_NAME not in _index_names(seeded)
    assert before <= _index_names(seeded)
    assert _observation_count(seeded) == (1,)


def test_coval_ingest_index_recovers_after_cancelled_concurrent_build(seeded: Any) -> None:
    before = _index_names(seeded)
    blocker = psycopg.connect(_dsn(seeded), autocommit=False)
    builder: psycopg.Connection[Any] | None = None
    try:
        blocker.execute(
            "UPDATE benchmarks_v2.benchmark_observations "
            "SET sample_id = 'blocked/simulation' WHERE sample_id = 'external/simulation'"
        )
        builder = psycopg.connect(_dsn(seeded), autocommit=True)
        builder.execute("SET statement_timeout = '250ms'")
        with pytest.raises(psycopg.errors.QueryCanceled):
            builder.execute(
                "CREATE INDEX CONCURRENTLY "
                f"{_INDEX_NAME} ON benchmarks_v2.benchmark_observations "
                "(provider, benchmark, (split_part(sample_id, '/', 1))) INCLUDE (id, run_id)"
            )
        valid, ready, full, key_count, _, _ = _index_info(builder)
        assert valid is False
        assert ready is False
        assert full is True
        assert key_count == 3
        assert seeded.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "20260918_0040",
        )
    finally:
        if builder is not None:
            builder.close()
        blocker.rollback()
        blocker.close()

    _migrate(seeded, "20260921_0041")
    _assert_valid_index(seeded)
    assert before <= _index_names(seeded)
    assert _observation_count(seeded) == (1,)


def test_coval_ingest_index_replaces_wrong_same_name_index(seeded: Any) -> None:
    before = _index_names(seeded)
    seeded.execute(f"CREATE INDEX {_INDEX_NAME} ON benchmarks_v2.benchmark_observations (provider)")

    _migrate(seeded, "20260921_0041")
    _assert_valid_index(seeded)
    assert before <= _index_names(seeded)
    assert _observation_count(seeded) == (1,)
