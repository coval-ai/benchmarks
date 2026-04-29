# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the Coval Benchmarks API test suite.

Uses ``pytest-postgresql`` to spin up a real in-process Postgres for each test.
The FastAPI lifespan is managed by ``asgi_lifespan.LifespanManager`` so that the
psycopg3 pool is properly opened and closed for each test.

Tests use ``httpx.AsyncClient`` with ``ASGITransport`` — no real network calls.

Design note on the pool singleton:
``coval_bench.db.conn.get_pool`` is a module-level singleton.  After a test's
pool is closed, the singleton is in a closed state and cannot be reopened.  To
keep tests isolated we patch the ``lifespan_pool`` function to create a fresh
pool per test instead of reusing the singleton.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import psycopg
import psycopg.rows
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from psycopg_pool import AsyncConnectionPool

from coval_bench.api.app import create_app
from coval_bench.config import Settings


def _make_db_url(postgresql: Any) -> str:
    """Build a postgresql:// URL from the pytest-postgresql fixture."""
    info = postgresql.info
    return f"postgresql://{info.user}:{info.password or ''}@{info.host}:{info.port}/{info.dbname}"


async def _apply_schema(dsn: str) -> None:
    """Create the benchmarks_v2 schema and tables needed by the API tests.

    This mirrors the Alembic migration (20260429_0001_init_schema) without
    requiring a full Alembic migration run.
    """
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        await aconn.execute("CREATE SCHEMA IF NOT EXISTS benchmarks_v2")
        await aconn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks_v2.runs (
                id             bigserial PRIMARY KEY,
                started_at     timestamptz NOT NULL DEFAULT now(),
                finished_at    timestamptz,
                runner_sha     text NOT NULL,
                dataset_id     text NOT NULL,
                dataset_sha256 text NOT NULL,
                status         text NOT NULL
                    CHECK (status IN ('running','succeeded','partial','failed')),
                error          text
            )
        """)
        await aconn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks_v2.results (
                id             bigserial PRIMARY KEY,
                run_id         bigint NOT NULL
                    REFERENCES benchmarks_v2.runs(id) ON DELETE CASCADE,
                provider       text NOT NULL,
                model          text NOT NULL,
                voice          text,
                benchmark      text NOT NULL CHECK (benchmark IN ('STT','TTS')),
                metric_type    text NOT NULL,
                metric_value   double precision,
                metric_units   text,
                audio_filename text,
                transcript     text,
                status         text NOT NULL CHECK (status IN ('success','failed')),
                error          text,
                created_at     timestamptz NOT NULL DEFAULT now()
            )
        """)
        # Materialized view for 24h leaderboard
        await aconn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS benchmarks_v2.results_24h AS
            SELECT provider, model, benchmark, metric_type,
                   AVG(metric_value)::float8 AS avg_value,
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY metric_value)::float8 AS p50,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY metric_value)::float8 AS p95,
                   COUNT(*) AS n
            FROM benchmarks_v2.results
            WHERE status = 'success'
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY provider, model, benchmark, metric_type
        """)
    finally:
        await aconn.close()


@pytest_asyncio.fixture
async def app(postgresql: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FastAPI]:
    """FastAPI app fixture with a real in-process Postgres.

    Creates a fresh pool per test (bypassing the module-level singleton) to
    ensure tests are fully isolated.
    """
    dsn = _make_db_url(postgresql)
    await _apply_schema(dsn)

    monkeypatch.setenv("DATABASE_URL", dsn)
    monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
    monkeypatch.setenv("DATASET_ID", "librispeech-test-clean-50")
    monkeypatch.setenv("RUNNER_SHA", "test-sha")

    settings = Settings()

    # Patch lifespan_pool in app.py to always create a fresh pool, bypassing
    # the module-level singleton which cannot be reopened once closed.
    PoolType = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]

    @asynccontextmanager
    async def fresh_lifespan_pool(s: Settings) -> AsyncIterator[PoolType]:
        pool: PoolType = AsyncConnectionPool(
            conninfo=str(s.database_url),
            min_size=1,
            max_size=2,
            open=False,
            kwargs={"autocommit": False, "row_factory": psycopg.rows.dict_row},
        )
        await pool.open()
        try:
            yield pool
        finally:
            await pool.close()

    monkeypatch.setattr("coval_bench.api.app.lifespan_pool", fresh_lifespan_pool)

    test_app = create_app(settings)
    async with LifespanManager(test_app):
        yield test_app


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """httpx AsyncClient pointed at the test FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


async def _insert_run(postgresql: Any, **kwargs: Any) -> int:
    """Helper: insert a run row and return its id."""
    dsn = _make_db_url(postgresql)
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        defaults: dict[str, Any] = {
            "runner_sha": "abc123",
            "dataset_id": "librispeech-test-clean-50",
            "dataset_sha256": "sha256test",
            "status": "succeeded",
        }
        defaults.update(kwargs)
        row = await aconn.execute(
            """
            INSERT INTO benchmarks_v2.runs (runner_sha, dataset_id, dataset_sha256, status)
            VALUES (%(runner_sha)s, %(dataset_id)s, %(dataset_sha256)s, %(status)s)
            RETURNING id
            """,
            defaults,
        )
        result = await row.fetchone()
        assert result is not None
        return int(result[0])
    finally:
        await aconn.close()


async def _insert_result(
    postgresql: Any,
    run_id: int,
    *,
    created_at: datetime | None = None,
    **kwargs: Any,
) -> int:
    """Helper: insert a result row and return its id.

    Pass ``created_at`` as a :class:`datetime` to override the DB default
    (``now()``).  This is useful for seeding rows at specific points in time
    for window-filter tests.  All other columns can be overridden via kwargs.
    """
    dsn = _make_db_url(postgresql)
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        defaults: dict[str, Any] = {
            "run_id": run_id,
            "provider": "deepgram",
            "model": "nova-3",
            "voice": None,
            "benchmark": "STT",
            "metric_type": "WER",
            "metric_value": 3.5,
            "metric_units": "%",
            "audio_filename": "test.wav",
            "status": "success",
        }
        defaults.update(kwargs)

        if created_at is not None:
            defaults["created_at"] = created_at
            row = await aconn.execute(
                """
                INSERT INTO benchmarks_v2.results
                    (run_id, provider, model, voice, benchmark, metric_type,
                     metric_value, metric_units, audio_filename, status, created_at)
                VALUES
                    (%(run_id)s, %(provider)s, %(model)s, %(voice)s, %(benchmark)s,
                     %(metric_type)s, %(metric_value)s, %(metric_units)s,
                     %(audio_filename)s, %(status)s, %(created_at)s)
                RETURNING id
                """,
                defaults,
            )
        else:
            row = await aconn.execute(
                """
                INSERT INTO benchmarks_v2.results
                    (run_id, provider, model, voice, benchmark, metric_type,
                     metric_value, metric_units, audio_filename, status)
                VALUES
                    (%(run_id)s, %(provider)s, %(model)s, %(voice)s, %(benchmark)s,
                     %(metric_type)s, %(metric_value)s, %(metric_units)s,
                     %(audio_filename)s, %(status)s)
                RETURNING id
                """,
                defaults,
            )
        result = await row.fetchone()
        assert result is not None
        return int(result[0])
    finally:
        await aconn.close()


async def _refresh_mv(postgresql: Any) -> None:
    """Refresh the results_24h materialized view."""
    dsn = _make_db_url(postgresql)
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        await aconn.execute("REFRESH MATERIALIZED VIEW benchmarks_v2.results_24h")
    finally:
        await aconn.close()
