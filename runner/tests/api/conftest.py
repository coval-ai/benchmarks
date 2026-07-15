# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the Coval Benchmarks API test suite.

Uses ``pytest-postgresql`` with a session-scoped in-process Postgres.  The
schema is loaded into the template database once; each test gets a fresh
database cloned from it.  The FastAPI lifespan is managed by
``asgi_lifespan.LifespanManager`` so that the psycopg3 pool is properly opened
and closed for each test.

Tests use ``httpx.AsyncClient`` with ``ASGITransport`` — no real network calls.

Design note on the pool singleton:
``coval_bench.db.conn.get_pool`` is a module-level singleton.  After a test's
pool is closed, the singleton is in a closed state and cannot be reopened.  To
keep tests isolated we patch the ``lifespan_pool`` function to create a fresh
pool per test instead of reusing the singleton.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import psycopg
import psycopg.rows
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql import factories

from coval_bench.api.app import create_app
from coval_bench.config import Settings

ARENA_LABELER_KEY = "test-labeler-key"


def _make_db_url(postgresql: Any) -> str:
    """Build a postgresql:// URL from the pytest-postgresql fixture."""
    info = postgresql.info
    return f"postgresql://{info.user}:{info.password or ''}@{info.host}:{info.port}/{info.dbname}"


# Mirrors the per-window matview migrations (20260611_0005 + 20260715_0010).
_MV_WINDOWS: dict[str, str] = {
    "results_24h": "24 hours",
    "results_7d": "7 days",
    "results_30d": "30 days",
}

# Dataset attribution for aggregate rows (mirrors migration 20260715_0010).
_DATASET_CASE_SQL = "CASE WHEN r.benchmark = 'TTS' THEN 'tts-v1' ELSE rn.dataset_id END"


def _load_schema(**connect_kwargs: Any) -> None:
    """Create the benchmarks_v2 schema and tables needed by the API tests.

    Runs once against the template database; per-test databases are cloned
    from it.  Mirrors the Alembic migration (20260429_0001_init_schema)
    without requiring a full Alembic migration run.
    """
    with psycopg.connect(**connect_kwargs) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS benchmarks_v2")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks_v2.runs (
                id             bigserial PRIMARY KEY,
                started_at     timestamptz NOT NULL DEFAULT now(),
                finished_at    timestamptz,
                scheduled_at   timestamptz,
                runner_sha     text NOT NULL,
                dataset_id     text NOT NULL,
                dataset_sha256 text NOT NULL,
                status         text NOT NULL
                    CHECK (status IN ('running','succeeded','partial','failed')),
                error          text
            )
        """)
        conn.execute("""
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
        # Per-window stats materialized views (model_stats + leaderboard).
        # Mirrors migration 20260715_0010: per-dataset rows plus pooled rows
        # under the '__all__' sentinel.
        # S608 false-positive: name and interval come from the _MV_WINDOWS constant.
        for name, interval in _MV_WINDOWS.items():
            conn.execute(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS benchmarks_v2.{name} AS
                SELECT provider, model, benchmark, dataset_id, metric_type,
                       avg_value, stddev_value, min_value,
                       pct[1] AS p25, pct[2] AS p50, pct[3] AS p75,
                       pct[4] AS p90, pct[5] AS p95, pct[6] AS p99,
                       max_value, sample_count
                FROM (
                    SELECT r.provider, r.model, r.benchmark,
                           COALESCE({_DATASET_CASE_SQL}, '__all__') AS dataset_id,
                           r.metric_type,
                           AVG(r.metric_value)::float8 AS avg_value,
                           COALESCE(STDDEV_SAMP(r.metric_value), 0)::float8 AS stddev_value,
                           MIN(r.metric_value)::float8 AS min_value,
                           PERCENTILE_CONT(ARRAY[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
                               WITHIN GROUP (ORDER BY r.metric_value)::float8[] AS pct,
                           MAX(r.metric_value)::float8 AS max_value,
                           COUNT(*)::int AS sample_count
                    FROM benchmarks_v2.results r
                    JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
                    WHERE r.status = 'success'
                      AND rn.status IN ('succeeded', 'partial')
                      AND r.metric_value IS NOT NULL
                      AND r.created_at >= now() - INTERVAL '{interval}'
                    GROUP BY GROUPING SETS (
                        (r.provider, r.model, r.benchmark, r.metric_type, {_DATASET_CASE_SQL}),
                        (r.provider, r.model, r.benchmark, r.metric_type)
                    )
                ) stats
            """)  # noqa: S608
        # Series rollup table (mirrors migrations 20260611_0006 + 20260715_0010).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks_v2.results_by_bucket (
                provider      text NOT NULL,
                model         text NOT NULL,
                benchmark     text NOT NULL CHECK (benchmark IN ('STT','TTS','S2S')),
                dataset_id    text NOT NULL,
                metric_type   text NOT NULL,
                bucket_at     timestamptz NOT NULL,
                min_value     double precision NOT NULL,
                p25           double precision NOT NULL,
                p50           double precision NOT NULL,
                p75           double precision NOT NULL,
                max_value     double precision NOT NULL,
                value_sum     double precision NOT NULL,
                sample_count  integer NOT NULL,
                PRIMARY KEY (provider, model, benchmark, dataset_id, metric_type, bucket_at)
            )
        """)


postgresql_proc = factories.postgresql_proc(load=[_load_schema])
postgresql = factories.postgresql("postgresql_proc")


@pytest_asyncio.fixture
async def app(postgresql: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FastAPI]:
    """FastAPI app fixture with a real in-process Postgres.

    Creates a fresh pool per test (bypassing the module-level singleton) to
    ensure tests are fully isolated.
    """
    dsn = _make_db_url(postgresql)

    monkeypatch.setenv("DATABASE_URL", dsn)
    monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
    monkeypatch.setenv("DATASET_ID", "librispeech-test-clean-50")
    monkeypatch.setenv("RUNNER_SHA", "test-sha")
    monkeypatch.setenv("POSTHOG_DISABLED", "true")
    monkeypatch.setenv("ARENA_LABELER_KEY", ARENA_LABELER_KEY)

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


@pytest.fixture
def app_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[dict[str, str] | None], Awaitable[FastAPI]]:
    """Build an app with a stubbed pool so lifespan and analytics wiring can be
    tested without spinning up Postgres. The caller drives the lifespan.
    """

    async def _factory(extra_env: dict[str, str] | None = None) -> FastAPI:
        monkeypatch.setenv("DATABASE_URL", "postgresql://runner:password@localhost:5432/benchmarks")
        monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
        monkeypatch.setenv("DATASET_ID", "stt-v1")
        monkeypatch.setenv("RUNNER_SHA", "test-sha")
        monkeypatch.setenv("POSTHOG_DISABLED", "true")
        for key, value in (extra_env or {}).items():
            monkeypatch.setenv(key, value)

        @asynccontextmanager
        async def stub_lifespan_pool(s: Settings) -> AsyncIterator[MagicMock]:
            yield MagicMock()

        monkeypatch.setattr("coval_bench.api.app.lifespan_pool", stub_lifespan_pool)
        return create_app(Settings())

    return _factory


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
            "scheduled_at": None,
        }
        defaults.update(kwargs)
        row = await aconn.execute(
            """
            INSERT INTO benchmarks_v2.runs
                (runner_sha, dataset_id, dataset_sha256, status, scheduled_at)
            VALUES
                (%(runner_sha)s, %(dataset_id)s, %(dataset_sha256)s, %(status)s,
                 %(scheduled_at)s)
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
    """Refresh all per-window stats materialized views."""
    dsn = _make_db_url(postgresql)
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        for name in _MV_WINDOWS:
            await aconn.execute(f"REFRESH MATERIALIZED VIEW benchmarks_v2.{name}")
    finally:
        await aconn.close()


# Scheduler period for the legacy created_at bucket fallback (matches
# migration 20260611_0006).
_BUCKET_PERIOD_SECONDS = 1800


async def _fill_buckets(postgresql: Any) -> None:
    """Truncate and recompute results_by_bucket from all results (mirrors the backfill)."""
    dsn = _make_db_url(postgresql)
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    bucket_sql = (
        "COALESCE(rn.scheduled_at, to_timestamp("
        f"floor(extract(epoch FROM r.created_at) / {_BUCKET_PERIOD_SECONDS})"
        f" * {_BUCKET_PERIOD_SECONDS}))"
    )
    try:
        await aconn.execute("TRUNCATE benchmarks_v2.results_by_bucket")
        await aconn.execute(f"""
            INSERT INTO benchmarks_v2.results_by_bucket
                (provider, model, benchmark, dataset_id, metric_type, bucket_at,
                 min_value, p25, p50, p75, max_value, value_sum, sample_count)
            SELECT r.provider, r.model, r.benchmark,
                   COALESCE({_DATASET_CASE_SQL}, '__all__'),
                   r.metric_type, {bucket_sql},
                   MIN(r.metric_value)::float8,
                   PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY r.metric_value)::float8,
                   PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY r.metric_value)::float8,
                   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.metric_value)::float8,
                   MAX(r.metric_value)::float8,
                   SUM(r.metric_value)::float8,
                   COUNT(*)::int
            FROM benchmarks_v2.results r
            JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
            WHERE r.status = 'success'
              AND rn.status IN ('succeeded', 'partial')
              AND r.metric_value IS NOT NULL
            GROUP BY GROUPING SETS (
                (r.provider, r.model, r.benchmark, r.metric_type, {_DATASET_CASE_SQL},
                 {bucket_sql}),
                (r.provider, r.model, r.benchmark, r.metric_type, {bucket_sql})
            )
        """)  # noqa: S608
    finally:
        await aconn.close()
