# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The seed migration against a real Postgres: every packaged rate valid, and the downgrade guard.

Recording semantics are covered over HTTP in ``tests/api/test_admin_pricing.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from pytest_postgresql.factories import postgresql

from coval_bench.db.pricing_store import PricingStore
from coval_bench.registries import Benchmark
from coval_bench.registries.pricing import NewRate

from .conftest import _INI_PATH, apply_migrations, async_dsn, open_pool

pricing_pg = postgresql("pg_proc")  # shared server from conftest, own per-test DB

_ADMIN = {"user_id": "user_cale", "email": "cale@coval.dev"}
KEY = (Benchmark.STT, "deepgram", "nova-3")
TODAY = datetime.now(UTC).date()


def _run(
    conn: psycopg.Connection[Any], scenario: Callable[[PricingStore], Awaitable[None]]
) -> None:
    apply_migrations(conn)

    async def go() -> None:
        pool = await open_pool(conn)
        try:
            await scenario(PricingStore(pool))
        finally:
            await pool.close()

    asyncio.run(go())


def _rate(price: str | None = "0.0050", **overrides: Any) -> NewRate:
    fields: dict[str, Any] = {
        "benchmark": KEY[0],
        "provider": KEY[1],
        "model": KEY[2],
        "unit": None if price is None else "per_minute",
        "price_usd": price,
        "effective_from": TODAY,
        "source_url": None if price is None else "https://deepgram.com/pricing",
    }
    return NewRate.model_validate({**fields, **overrides})


def test_the_migration_seeds_valid_rates_on_registered_models(
    pricing_pg: psycopg.Connection[Any],
) -> None:
    async def scenario(store: PricingStore) -> None:
        recordings = await store.recordings()
        assert len(recordings) == 61 == len({r.key for r in recordings})
        for r in recordings:
            NewRate(
                benchmark=r.benchmark,
                provider=r.provider,
                model=r.model,
                unit=r.unit,
                price_usd=r.price_usd,
                effective_from=r.effective_from,
                source_url=r.source_url,
                notes=r.notes,
            )
            assert r.priced and r.recorded_by_user_id == "migration:20260903_0027"
            assert await store.model_exists(r.key), r.key
        by_key = {r.key: r for r in recordings}
        # The quoted scale survives the round trip: 0.030, not 0.03.
        assert str(by_key[KEY].price_usd) == "0.0048"
        assert str(by_key[(Benchmark.TTS, "deepgram", "aura-2-thalia-en")].price_usd) == "0.030"

    _run(pricing_pg, scenario)


def _downgrade_one(conn: psycopg.Connection[Any]) -> None:
    cfg = AlembicConfig(str(_INI_PATH))
    cfg.set_main_option(
        "sqlalchemy.url", async_dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    alembic_command.downgrade(cfg, "-1")


def test_downgrade_drops_only_a_log_that_holds_nothing_beyond_the_seed(
    pricing_pg: psycopg.Connection[Any],
) -> None:
    async def scenario(store: PricingStore) -> None:
        await store.record(_rate("0.0061"), **_ADMIN)

    _run(pricing_pg, scenario)
    with pytest.raises(RuntimeError, match="refusing to drop"):
        _downgrade_one(pricing_pg)
    with pricing_pg.cursor() as cur:
        cur.execute("DELETE FROM benchmarks_v2.pricing_rates WHERE price_usd = 0.0061")
        pricing_pg.commit()
    _downgrade_one(pricing_pg)
    with pricing_pg.cursor() as cur:
        cur.execute("SELECT to_regclass('benchmarks_v2.pricing_rates')")
        assert cur.fetchone() == (None,)
