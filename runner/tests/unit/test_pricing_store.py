# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The pricing log's two doors, against a real Postgres brought up by the migrations.

Running the migrations here is the seed's parity test: after ``upgrade head``
the log must hold exactly the packaged ratesheets, so a ratesheet edit that
forgets the ``SEED_ROWS`` literal (or the reverse) fails right here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import psycopg
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from pydantic import ValidationError
from pytest_postgresql.factories import postgresql

from coval_bench.db.pricing_store import NewRate, PricingStore
from coval_bench.registries import Benchmark
from coval_bench.registries.pricing.resolve import timelines
from coval_bench.registries.pricing.schema import PricingUnit

from .conftest import _INI_PATH, apply_migrations, async_dsn, open_pool

pricing_pg = postgresql("pg_proc")  # shared server from conftest, own per-test DB

_ADMIN = {"user_id": "user_cale", "email": "cale@coval.dev"}
SEEDED_KEY = (Benchmark.STT, "deepgram", "nova-3")
TODAY = datetime.now(UTC).date()


def _with_store(
    conn: psycopg.Connection[Any],
    scenario: Callable[[PricingStore, psycopg.Connection[Any]], Awaitable[None]],
) -> None:
    apply_migrations(conn)

    async def _run() -> None:
        pool = await open_pool(conn)
        try:
            await scenario(PricingStore(pool), conn)
        finally:
            await pool.close()

    asyncio.run(_run())


def _rate(price: str | None = "0.0050", **overrides: Any) -> NewRate:
    fields: dict[str, Any] = {
        "benchmark": SEEDED_KEY[0],
        "provider": SEEDED_KEY[1],
        "model": SEEDED_KEY[2],
        "unit": None if price is None else PricingUnit.PER_MINUTE,
        "price_usd": price,
        "effective_from": TODAY,
        "source_url": None if price is None else "https://deepgram.com/pricing",
    }
    fields.update(overrides)
    return NewRate.model_validate(fields)


def test_the_migration_seeds_valid_rates_on_registered_models(
    pricing_pg: psycopg.Connection[Any],
) -> None:
    """Every seeded row passes the same rules an admin write must, and prices a real model."""

    async def scenario(store: PricingStore, conn: psycopg.Connection[Any]) -> None:
        recordings = await store.recordings()
        assert len(recordings) == 61
        assert len({r.key for r in recordings}) == 61  # one rate per model
        for recording in recordings:
            NewRate(
                benchmark=recording.key[0],
                provider=recording.key[1],
                model=recording.key[2],
                unit=recording.unit,
                price_usd=recording.price_usd,
                effective_from=recording.effective_from,
                source_url=recording.source_url,
                notes=recording.notes,
            )
            assert recording.priced, recording.key
            assert recording.recorded_by_user_id == "migration:20260903_0027"
            assert recording.recorded_by_email is None
            assert await store.model_exists(recording.key), recording.key
        # The scale of the quoted figure survives the round trip (0.030, not 0.03).
        aura = next(
            r for r in recordings if r.key == (Benchmark.TTS, "deepgram", "aura-2-thalia-en")
        )
        assert str(aura.price_usd) == "0.030"
        nova = next(r for r in recordings if r.key == SEEDED_KEY)
        assert str(nova.price_usd) == "0.0048"

    _with_store(pricing_pg, scenario)


def test_record_appends_ignores_repeats_and_keeps_corrections(
    pricing_pg: psycopg.Connection[Any],
) -> None:
    async def scenario(store: PricingStore, conn: psycopg.Connection[Any]) -> None:
        first, inserted = await store.record(_rate("0.0050"), **_ADMIN)
        assert inserted
        assert (first.recorded_by_user_id, first.recorded_by_email) == (
            "user_cale",
            "cale@coval.dev",
        )

        again, inserted = await store.record(_rate("0.0050"), **_ADMIN)
        assert not inserted and again.id == first.id

        fix, inserted = await store.record(_rate("0.0055"), **_ADMIN)
        assert inserted and fix.id != first.id
        timeline = timelines(await store.recordings_for(SEEDED_KEY))[SEEDED_KEY]
        current = timeline.in_force(TODAY)
        assert current is not None and current.recording.id == fix.id
        assert [r.id for r in timeline.superseded] == [first.id]
        # Yesterday still reads the seeded rate.
        seeded = timeline.in_force(TODAY - timedelta(days=1))
        assert seeded is not None and str(seeded.recording.price_usd) == "0.0048"

        # Going back to a value that lost is a real change, not a repeat.
        revert, inserted = await store.record(_rate("0.0050"), **_ADMIN)
        assert inserted and revert.id not in {first.id, fix.id}
        assert await store.count_for(SEEDED_KEY) == 4

    _with_store(pricing_pg, scenario)


def test_a_delisting_records_no_price(pricing_pg: psycopg.Connection[Any]) -> None:
    async def scenario(store: PricingStore, conn: psycopg.Connection[Any]) -> None:
        gone, inserted = await store.record(
            _rate(None, notes="Page no longer prints a rate."), **_ADMIN
        )
        assert inserted and not gone.priced
        assert gone.unit is None and gone.source_url is None
        timeline = timelines(await store.recordings_for(SEEDED_KEY))[SEEDED_KEY]
        current = timeline.in_force(TODAY)
        assert current is not None and not current.recording.priced
        assert not await store.model_exists((Benchmark.STT, "ghost", "nobody"))

    _with_store(pricing_pg, scenario)


def _downgrade_one(conn: psycopg.Connection[Any]) -> None:
    """Step the per-test database back one revision, off the pricing log."""
    cfg = AlembicConfig(str(_INI_PATH))
    cfg.set_main_option(
        "sqlalchemy.url", async_dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    alembic_command.downgrade(cfg, "-1")


def test_downgrade_refuses_to_forget_admin_recordings(pricing_pg: psycopg.Connection[Any]) -> None:
    """The log is the only home a recorded price has; a downgrade may not drop it."""

    async def scenario(store: PricingStore, conn: psycopg.Connection[Any]) -> None:
        await store.record(_rate("0.0061"), **_ADMIN)

    _with_store(pricing_pg, scenario)
    with pytest.raises(RuntimeError, match="refusing to drop"):
        _downgrade_one(pricing_pg)
    with pricing_pg.cursor() as cur:
        cur.execute("SELECT count(*) FROM benchmarks_v2.pricing_rates WHERE price_usd = 0.0061")
        assert cur.fetchone() == (1,)


def test_downgrade_drops_a_log_that_holds_only_the_seed(
    pricing_pg: psycopg.Connection[Any],
) -> None:
    apply_migrations(pricing_pg)
    _downgrade_one(pricing_pg)
    with pricing_pg.cursor() as cur:
        cur.execute("SELECT to_regclass('benchmarks_v2.pricing_rates')")
        assert cur.fetchone() == (None,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"price_usd": None},  # unit without a price
        {"unit": None},  # price without a unit
        {"price_usd": 0.005},  # a bare number would drop trailing zeros
        {"price_usd": "0"},
        {"unit": PricingUnit.PER_1K_CHARS},  # a character unit cannot bill STT
        {"source_url": None},  # a priced rate must cite its page
        {"source_url": "not a url"},
        {"provider": ""},
    ],
)
def test_new_rate_rejects_what_a_ratesheet_would(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _rate(**overrides)


def test_new_rate_strips_blank_notes_and_matches_as_written() -> None:
    rate = _rate(notes="   ")
    assert rate.notes is None
    assert _rate("0.20").price_usd == Decimal("0.20")
    assert str(_rate("0.20").price_usd) == "0.20"
