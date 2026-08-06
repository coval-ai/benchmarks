# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the append-only ``model_pricing`` store.

Same embedded-Postgres harness as ``test_db_writer``: real DB, migrations
applied per test, no remote connections.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
import psycopg.errors
import psycopg.rows
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql

from coval_bench.db.models import Benchmark, BillingUnit
from coval_bench.db.pricing import PricingStore
from coval_bench.registries.models import MODEL_REGISTRY, ModelStatus
from scripts.seed_pricing import RATESHEET, apply_ratesheet

pg_conn = postgresql("pg_proc")  # function-scoped clean DB per test

_INI_PATH = Path(__file__).parents[2] / "alembic.ini"


def _async_dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    host = info.host or "localhost"
    port = info.port or 5432
    user = info.user or ""
    password = f":{info.password}" if info.password else ""
    return f"postgresql://{user}{password}@{host}:{port}/{info.dbname or 'test'}"


def _apply_migrations(conn: psycopg.Connection[Any]) -> None:
    cfg = AlembicConfig(str(_INI_PATH))
    cfg.set_main_option(
        "sqlalchemy.url", _async_dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    alembic_command.upgrade(cfg, "head")


async def _make_pool(
    conn: psycopg.Connection[Any],
) -> AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]:
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]] = AsyncConnectionPool(
        conninfo=_async_dsn(conn),
        min_size=1,
        max_size=4,
        open=False,
        kwargs={"autocommit": False, "row_factory": psycopg.rows.dict_row},
    )
    await pool.open()
    return pool


_T0 = datetime(2026, 8, 1, tzinfo=UTC)


async def _upsert(store: PricingStore, rate: str, *, at: datetime, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "provider": "deepgram",
        "model": "nova-3",
        "benchmark": Benchmark.STT,
        "billing_unit": BillingUnit.PER_MINUTE,
        "rate_usd": Decimal(rate),
        "source_url": "https://deepgram.com/pricing",
        "as_of": date(2026, 8, 6),
        "updated_by": "human",
        "effective_at": at,
    }
    kwargs.update(overrides)
    return await store.upsert_rate(**kwargs)


def test_unique_effective_rate_enforced(pg_conn: psycopg.Connection[Any]) -> None:
    """The partial unique index rejects a second effective row for one unit."""
    _apply_migrations(pg_conn)
    pg_conn.autocommit = True
    insert = (
        "INSERT INTO benchmarks_v2.model_pricing "
        "(provider, model, benchmark, billing_unit, rate_usd, effective_at, "
        " source_url, as_of, updated_by) "
        "VALUES ('deepgram', 'nova-3', 'STT', 'per_minute', %s, now(), "
        " 'https://deepgram.com/pricing', current_date, 'human')"
    )
    with pg_conn.cursor() as cur:
        cur.execute(insert, ("0.0059",))
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(insert, ("0.0077",))


def test_supersede_flow_keeps_history(pg_conn: psycopg.Connection[Any]) -> None:
    """upsert supersedes atomically; a backdated query returns the old rate."""
    _apply_migrations(pg_conn)

    async def _run() -> None:
        pool = await _make_pool(pg_conn)
        try:
            store = PricingStore(pool)
            row_a, created_a = await _upsert(store, "0.0059", at=_T0)
            assert created_a and row_a.superseded_at is None

            # Identical re-upsert is a no-op (seed idempotency).
            row_same, created_same = await _upsert(store, "0.0059", at=_T0 + timedelta(days=1))
            assert not created_same
            assert row_same.id == row_a.id

            t1 = _T0 + timedelta(days=2)
            row_b, created_b = await _upsert(store, "0.0077", at=t1)
            assert created_b

            # History: before the change the old rate is effective, after it the new.
            old = await store.get_effective_rates("deepgram", "nova-3", _T0 + timedelta(days=1))
            assert [r.rate_usd for r in old] == [Decimal("0.0059")]
            assert old[0].superseded_at == t1
            new = await store.get_effective_rates("deepgram", "nova-3", t1)
            assert [r.id for r in new] == [row_b.id]
            # Before the first rate existed: nothing.
            before = await store.get_effective_rates("deepgram", "nova-3", _T0 - timedelta(days=1))
            assert before == []
        finally:
            await pool.close()

    asyncio.run(_run())


def test_token_billed_model_holds_two_effective_rows(pg_conn: psycopg.Connection[Any]) -> None:
    _apply_migrations(pg_conn)

    async def _run() -> None:
        pool = await _make_pool(pg_conn)
        try:
            store = PricingStore(pool)
            await _upsert(
                store,
                "6.0",
                at=_T0,
                model="gpt-4o-transcribe",
                billing_unit=BillingUnit.PER_1M_TOKENS_INPUT,
                provider="openai",
            )
            await _upsert(
                store,
                "10.0",
                at=_T0,
                model="gpt-4o-transcribe",
                billing_unit=BillingUnit.PER_1M_TOKENS_OUTPUT,
                provider="openai",
            )
            rates = await store.get_effective_rates("openai", "gpt-4o-transcribe", _T0)
            assert {r.billing_unit for r in rates} == {
                BillingUnit.PER_1M_TOKENS_INPUT,
                BillingUnit.PER_1M_TOKENS_OUTPUT,
            }

            await store.load_cache()
            cached = store.effective_rates_cached("openai", "gpt-4o-transcribe")
            assert len(cached) == 2
            assert store.effective_rates_cached("openai", "unknown") == []

            # The benchmark filter keeps shared model ids apart (gradium
            # serves STT and TTS under "default").
            await _upsert(store, "0.013", at=_T0, provider="gradium", model="default")
            await _upsert(
                store,
                "48.0",
                at=_T0,
                provider="gradium",
                model="default",
                benchmark=Benchmark.TTS,
                billing_unit=BillingUnit.PER_1M_CHARS,
            )
            both = await store.get_effective_rates("gradium", "default", _T0)
            assert len(both) == 2
            stt_only = await store.get_effective_rates(
                "gradium", "default", _T0, benchmark=Benchmark.STT
            )
            assert [r.benchmark for r in stt_only] == [Benchmark.STT]
            await store.load_cache()
            assert len(store.effective_rates_cached("gradium", "default")) == 2
            tts_only = store.effective_rates_cached("gradium", "default", Benchmark.TTS)
            assert [r.benchmark for r in tts_only] == [Benchmark.TTS]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_cache_requires_explicit_load() -> None:
    store = PricingStore(pool=None)  # type: ignore[arg-type] — never touched before load
    with pytest.raises(RuntimeError, match="load_cache"):
        store.effective_rates_cached("deepgram", "nova-3")


def test_ratesheet_keys_match_registry() -> None:
    """Every ratesheet entry maps to a real active model — no orphan prices."""
    active = {
        (m.benchmark, m.provider, m.model)
        for m in MODEL_REGISTRY
        if m.status in (ModelStatus.ACTIVE, ModelStatus.EARLY_ACCESS)
    }
    orphans = set(RATESHEET) - active
    assert not orphans, f"ratesheet entries for unknown/inactive models: {sorted(orphans)}"


def test_seed_is_idempotent(pg_conn: psycopg.Connection[Any]) -> None:
    """Second apply_ratesheet run writes nothing new."""
    _apply_migrations(pg_conn)

    async def _run() -> None:
        pool = await _make_pool(pg_conn)
        try:
            store = PricingStore(pool)
            inserted, unchanged, gaps = await apply_ratesheet(store)
            assert inserted > 0
            assert unchanged == 0
            inserted2, unchanged2, gaps2 = await apply_ratesheet(store)
            assert inserted2 == 0
            assert unchanged2 == inserted
            assert gaps2 == gaps
        finally:
            await pool.close()

    asyncio.run(_run())
