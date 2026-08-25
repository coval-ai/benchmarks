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

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import jwt
import psycopg
import psycopg.rows
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql import factories

from coval_bench.api.app import create_app
from coval_bench.arena.moderation import ModerationResult
from coval_bench.arena.pairing import active_tts_models
from coval_bench.config import Settings
from coval_bench.registries import MODEL_REGISTRY
from coval_bench.registries.provider_keys import PROVIDER_ENV

ARENA_LABELER_KEY = "test-labeler-key"

# The one early-access proof is a Clerk session token, so the app fixture wires a
# whole stub instance: an issuer, an authorized party, a signing key the stubbed
# JWKS lookup resolves, and org grants. Two synthetic embargoed models sit on one
# provider, with an org entitled to each — same provider on purpose: it proves the
# grants separate callers by model, not just by vendor.
CLERK_ISSUER = "https://clerk.test.example.com"
CLERK_PARTY = "https://benchmarks.test.example.com"
COVAL_ORG = "org_test_coval"

EA_PROVIDER = "acme"
EA_MODEL = "unreleased-stt"
EA_MODEL_OTHER = "unreleased-stt-2"
EA_ORG = "org_test_first"
EA_ORG_OTHER = "org_test_second"
CLERK_ORG_PROVIDERS = {
    EA_ORG: [f"{EA_PROVIDER}/{EA_MODEL}"],
    EA_ORG_OTHER: [f"{EA_PROVIDER}/{EA_MODEL_OTHER}"],
}

_CLERK_SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_CLERK_PUBLIC_PEM = _CLERK_SIGNING_KEY.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)


def mint_clerk_token(**claims: Any) -> str:
    """A session token the stubbed Clerk instance vouches for."""
    now = int(time.time())
    payload: dict[str, Any] = {"iss": CLERK_ISSUER, "iat": now, "exp": now + 60, "azp": CLERK_PARTY}
    payload.update(claims)
    return jwt.encode(payload, _CLERK_SIGNING_KEY, algorithm="RS256")


def bearer(**claims: Any) -> dict[str, str]:
    """Request headers proving whatever *claims* say (e.g. ``email=``, ``org_id=``)."""
    return {"Authorization": f"Bearer {mint_clerk_token(**claims)}"}


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
                benchmark      text NOT NULL CHECK (benchmark IN ('STT','TTS','S2S')),
                metric_type    text NOT NULL,
                metric_value   double precision,
                metric_units   text,
                audio_filename text,
                transcript     text,
                status         text NOT NULL CHECK (status IN ('success','failed')),
                error          text,
                created_at     timestamptz NOT NULL DEFAULT now(),
                wer_insertions_pct    double precision,
                wer_deletions_pct     double precision,
                wer_substitutions_pct double precision
            )
        """)
        # Per-window stats materialized views (model_stats + leaderboard).
        # Mirrors migration 20260715_0010: per-dataset rows plus pooled rows
        # under the '__all__' sentinel, and 20260804_0014's WER breakdown.
        # S608 false-positive: name and interval come from the _MV_WINDOWS constant.
        for name, interval in _MV_WINDOWS.items():
            conn.execute(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS benchmarks_v2.{name} AS
                SELECT provider, model, benchmark, dataset_id, metric_type,
                       avg_value, stddev_value, min_value,
                       pct[1] AS p25, pct[2] AS p50, pct[3] AS p75,
                       pct[4] AS p90, pct[5] AS p95, pct[6] AS p99,
                       max_value, sample_count,
                       wer_insertions_pct, wer_deletions_pct, wer_substitutions_pct
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
                           COUNT(*)::int AS sample_count,
                           CASE WHEN COUNT(r.wer_insertions_pct) = COUNT(*)
                               AND COUNT(r.wer_deletions_pct) = COUNT(*)
                               AND COUNT(r.wer_substitutions_pct) = COUNT(*)
                               THEN AVG(r.wer_insertions_pct)::float8 END AS wer_insertions_pct,
                           CASE WHEN COUNT(r.wer_insertions_pct) = COUNT(*)
                               AND COUNT(r.wer_deletions_pct) = COUNT(*)
                               AND COUNT(r.wer_substitutions_pct) = COUNT(*)
                               THEN AVG(r.wer_deletions_pct)::float8 END AS wer_deletions_pct,
                           CASE WHEN COUNT(r.wer_insertions_pct) = COUNT(*)
                               AND COUNT(r.wer_deletions_pct) = COUNT(*)
                               AND COUNT(r.wer_substitutions_pct) = COUNT(*)
                               THEN AVG(r.wer_substitutions_pct)::float8
                               END AS wer_substitutions_pct
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
        # Model/tag registry tables (mirrors migration 20260824_0020).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks_v2.models (
                id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                modality      text NOT NULL CHECK (modality IN ('STT','TTS','S2S')),
                provider      text NOT NULL CHECK (provider <> ''),
                model         text NOT NULL CHECK (model <> ''),
                voice         text CHECK (voice IS NULL OR voice <> ''),
                voices        jsonb NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(voices) = 'array'),
                creator       text CHECK (creator IS NULL OR creator <> ''),
                source        text NOT NULL DEFAULT 'official-api' CHECK (source <> ''),
                licensing     text NOT NULL DEFAULT 'proprietary' CHECK (licensing <> ''),
                on_prem       boolean NOT NULL DEFAULT false,
                region        text CHECK (region IN ('us','eu','asia')),
                arena_enabled boolean NOT NULL DEFAULT true,
                collected     boolean NOT NULL,
                published     boolean NOT NULL,
                updated_by_user_id text NOT NULL CHECK (updated_by_user_id <> ''),
                updated_by_email   text CHECK (updated_by_email IS NULL OR updated_by_email <> ''),
                updated_at    timestamptz NOT NULL DEFAULT now(),
                UNIQUE (modality, provider, model)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks_v2.tags (
                value    text PRIMARY KEY CHECK (value <> ''),
                category text NOT NULL CHECK (category IN ('mode','features')),
                label    text NOT NULL CHECK (label <> '')
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks_v2.model_tags (
                model_id bigint NOT NULL REFERENCES benchmarks_v2.models(id) ON DELETE CASCADE,
                tag      text NOT NULL REFERENCES benchmarks_v2.tags(value),
                PRIMARY KEY (model_id, tag)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks_v2.model_history (
                id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                model_id  bigint NOT NULL,
                modality  text NOT NULL CHECK (modality IN ('STT','TTS','S2S')),
                provider  text NOT NULL CHECK (provider <> ''),
                model     text NOT NULL CHECK (model <> ''),
                old       jsonb CHECK (old IS NULL OR jsonb_typeof(old) = 'object'),
                new       jsonb NOT NULL CHECK (jsonb_typeof(new) = 'object'),
                changed_by_user_id text NOT NULL CHECK (changed_by_user_id <> ''),
                changed_by_org_id  text
                    CHECK (changed_by_org_id IS NULL OR changed_by_org_id <> ''),
                changed_by_email   text CHECK (changed_by_email IS NULL OR changed_by_email <> ''),
                changed_at timestamptz NOT NULL DEFAULT now()
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS model_history_model_id_changed_at
                ON benchmarks_v2.model_history (model_id, changed_at DESC)
        """)


postgresql_proc = factories.postgresql_proc(load=[_load_schema])
postgresql = factories.postgresql("postgresql_proc")


@pytest_asyncio.fixture
async def app(
    postgresql: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> AsyncIterator[FastAPI]:
    """FastAPI app fixture with a real in-process Postgres.

    Creates a fresh pool per test (bypassing the module-level singleton) to
    ensure tests are fully isolated.
    """
    dsn = _make_db_url(postgresql)

    # The default arena_audio_dir is CWD-relative: clips would land in the repo.
    monkeypatch.setenv("ARENA_AUDIO_DIR", str(tmp_path / "arena-audio"))

    monkeypatch.setenv("DATABASE_URL", dsn)
    monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
    monkeypatch.setenv("DATASET_ID", "librispeech-test-clean-50")
    monkeypatch.setenv("POSTHOG_DISABLED", "true")
    monkeypatch.setenv("ARENA_LABELER_KEY", ARENA_LABELER_KEY)
    monkeypatch.setenv("CLERK_ISSUER", CLERK_ISSUER)
    monkeypatch.setenv("CLERK_AUTHORIZED_PARTIES", json.dumps([CLERK_PARTY]))
    monkeypatch.setenv("CLERK_ORG_PROVIDERS", json.dumps(CLERK_ORG_PROVIDERS))
    monkeypatch.setenv("CLERK_COVAL_ORG", COVAL_ORG)
    # Resolve token signatures against the fixture key instead of the network.
    signing_key = SimpleNamespace(key=_CLERK_PUBLIC_PEM)
    monkeypatch.setattr(
        "coval_bench.api.clerk._jwks",
        lambda issuer: SimpleNamespace(get_signing_key_from_jwt=lambda token: signing_key),
    )
    # Battle generation screens prompts through the moderation API. Without this the
    # suite would reach the network on any machine that has the key exported.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Arena pairing drops providers whose key is not configured, so a service with no
    # keys has no roster and every battle is a 503. Prod mounts all of them — CI's
    # check_arena_keys.py enforces it — so the fixture models that. OPENAI_API_KEY stays
    # unset on purpose above, which simply leaves openai out of the roster.
    for model in active_tts_models(MODEL_REGISTRY):
        env_var = PROVIDER_ENV.get(model.provider)
        if env_var is not None and env_var != "OPENAI_API_KEY":
            monkeypatch.setenv(env_var, "test-provider-key")

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


@pytest.fixture(autouse=True)
def moderation_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to a reachable moderator that flags nothing.

    Battle generation fails closed when moderation is unavailable, and the suite has no
    API key, so without this every create-battle test would 503 regardless of subject.
    Tests about screening override it.
    """

    async def _clean(*args: Any, **kwargs: Any) -> ModerationResult:
        return ModerationResult(flagged=False, available=True)

    monkeypatch.setattr("coval_bench.api.routers.arena.moderation_verdict", _clean)


@pytest.fixture
def app_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[str, str] | None], Awaitable[FastAPI]]:
    """Build an app with a stubbed pool so lifespan and analytics wiring can be
    tested without spinning up Postgres. The caller drives the lifespan.
    """

    async def _factory(extra_env: dict[str, str] | None = None) -> FastAPI:
        monkeypatch.setenv("DATABASE_URL", "postgresql://runner:password@localhost:5432/benchmarks")
        monkeypatch.setenv("DATASET_BUCKET", "test-bucket")
        monkeypatch.setenv("DATASET_ID", "stt-v1")
        monkeypatch.setenv("POSTHOG_DISABLED", "true")
        monkeypatch.setenv("ARENA_AUDIO_DIR", str(tmp_path / "arena-audio"))
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

        # S608 false-positive: column names are dict keys set here, never caller values.
        columns = ", ".join(defaults)
        placeholders = ", ".join(f"%({c})s" for c in defaults)
        row = await aconn.execute(
            f"INSERT INTO benchmarks_v2.results ({columns})"  # noqa: S608
            f" VALUES ({placeholders}) RETURNING id",
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
