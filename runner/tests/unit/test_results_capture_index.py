# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone PostgreSQL checks for the v2 results capture-order index."""

from __future__ import annotations

from typing import Any

import pytest
from pytest_postgresql.factories import postgresql

from tests.unit.test_normalized_metric_id_migration import _migrate

pg_conn = postgresql("pg_proc")


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


def test_capture_order_index_is_full_ordered_and_reversible(seeded: Any) -> None:
    before = _index_names(seeded)
    assert seeded.execute(
        "SELECT count(*) FROM benchmarks_v2.benchmark_observations WHERE sample_id = 'sample'"
    ).fetchone() == (1,)

    _migrate(seeded, "20260917_0038")
    index_row = seeded.execute(
        """
        SELECT i.indisvalid, i.indpred IS NULL, pg_get_indexdef(i.indexrelid)
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'benchmarks_v2'
          AND c.relname = 'benchmark_observations_capture_order_idx'
        """
    ).fetchone()
    assert index_row is not None
    assert index_row[0] is True
    assert index_row[1] is True
    assert "(captured_at DESC, id DESC)" in index_row[2]
    assert before <= _index_names(seeded)
    assert seeded.execute(
        "SELECT count(*) FROM benchmarks_v2.benchmark_observations WHERE sample_id = 'sample'"
    ).fetchone() == (1,)

    _migrate(seeded, "20260916_0037", down=True)
    assert "benchmark_observations_capture_order_idx" not in _index_names(seeded)
    assert before <= _index_names(seeded)
    assert seeded.execute(
        "SELECT count(*) FROM benchmarks_v2.benchmark_observations WHERE sample_id = 'sample'"
    ).fetchone() == (1,)
