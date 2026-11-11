# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL checks for probe_results and its writer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db.probe_store import ProbeResult, insert_probe_result
from tests.unit.conftest import apply_migrations, open_pool

pg_conn = postgresql("pg_proc")

_TICK = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


def _probe(**fields: Any) -> ProbeResult:
    defaults: dict[str, Any] = {
        "benchmark": "TTS",
        "provider": "acme",
        "model": "fast-1",
        "scheduled_at": _TICK,
        "sent_at": _TICK,
        "ttfa_ms": 180.0,
    }
    return ProbeResult(**{**defaults, **fields})


def _rows(conn: Any) -> list[Any]:
    return list(
        conn.execute(
            "SELECT model, status, ttfa_ms, error, failure_origin"
            " FROM benchmarks_v2.probe_results ORDER BY id"
        ).fetchall()
    )


async def test_stores_a_success_and_a_failure(pg_conn: Any) -> None:
    apply_migrations(pg_conn)
    pool = await open_pool(pg_conn)
    try:
        await insert_probe_result(pool, _probe())
        await insert_probe_result(
            pool,
            _probe(model="slow-1", ttfa_ms=None, error="timed out", failure_origin="provider"),
        )
    finally:
        await pool.close()

    pg_conn.autocommit = True
    assert _rows(pg_conn) == [
        ("fast-1", "succeeded", 180.0, None, None),
        ("slow-1", "failed", None, "timed out", "provider"),
    ]


async def test_a_retried_tick_keeps_the_first_row(pg_conn: Any) -> None:
    apply_migrations(pg_conn)
    pool = await open_pool(pg_conn)
    try:
        await insert_probe_result(pool, _probe(ttfa_ms=180.0))
        await insert_probe_result(pool, _probe(ttfa_ms=950.0))
    finally:
        await pool.close()

    pg_conn.autocommit = True
    assert [row[2] for row in _rows(pg_conn)] == [180.0]


def test_a_failure_needs_an_error_and_an_origin(pg_conn: Any) -> None:
    apply_migrations(pg_conn)
    pg_conn.autocommit = True
    with pytest.raises(psycopg.errors.CheckViolation):
        pg_conn.execute(
            """
            INSERT INTO benchmarks_v2.probe_results
              (benchmark, provider, model, scheduled_at, sent_at, status)
            VALUES ('STT', 'acme', 'fast-1', now(), now(), 'failed')
            """
        )
