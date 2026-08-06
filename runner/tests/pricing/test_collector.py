# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Price-collector tests: fixture pages, mocked LLM, real Postgres gate checks."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import psycopg
import psycopg.rows
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql

from coval_bench.config import Settings
from coval_bench.db.models import Benchmark, BillingUnit
from coval_bench.db.pricing import PricingStore
from coval_bench.pricing.collector import (
    ExtractedRate,
    _match_model,
    _stale_rates,
    _store_snapshot,
    run_update_prices,
)
from coval_bench.registries.models import MODEL_REGISTRY
from coval_bench.registries.pricing_sources import PRICING_SOURCES

pg_conn = postgresql("pg_proc")

_FIXTURES = Path(__file__).parent / "fixtures"
_INI_PATH = Path(__file__).parents[2] / "alembic.ini"

_SETTINGS = Settings(
    database_url="postgresql://runner:password@localhost:5432/benchmarks",
    dataset_bucket="test-bucket",
    dataset_id="stt-v1",
    runner_sha="test-sha",
    posthog_disabled=True,
)


def _async_dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    password = f":{info.password}" if info.password else ""
    return (
        f"postgresql://{info.user or ''}{password}@{info.host or 'localhost'}:"
        f"{info.port or 5432}/{info.dbname or 'test'}"
    )


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


def _fixture_page(name: str) -> str:
    return (_FIXTURES / name).read_text()


async def _seed_nova3(store: PricingStore, rate: str = "0.0048") -> None:
    await store.upsert_rate(
        provider="deepgram",
        model="nova-3",
        benchmark=Benchmark.STT,
        billing_unit=BillingUnit.PER_MINUTE,
        rate_usd=Decimal(rate),
        source_url="https://deepgram.com/pricing",
        as_of=date(2026, 8, 6),
        updated_by="human",
    )


def _extracted(rate: float, *, model: str = "Nova-3 Monolingual", confidence: str = "high") -> Any:
    return ExtractedRate(
        model=model,
        billing_unit=BillingUnit.PER_MINUTE,
        rate_usd=rate,
        confidence=confidence,  # type: ignore[arg-type]
        quote=f"Nova-3 Monolingual ${rate}/min",
    )


def _run_collector(
    pg: psycopg.Connection[Any],
    rates: list[Any],
    *,
    seed_rate: str | None = "0.0048",
    dry_run: bool = False,
    fetch_side_effect: Any = None,
) -> tuple[dict[str, Any], list[tuple[str, str]], PricingStore]:
    """Drive run_update_prices for deepgram only, LLM + fetch + reviews mocked."""
    reviews: list[tuple[str, str]] = []

    async def _capture_review(settings: Any, title: str, body: str) -> None:
        reviews.append((title, body))

    async def _run() -> tuple[dict[str, Any], PricingStore]:
        pool = await _make_pool(pg)
        try:
            store = PricingStore(pool)
            if seed_rate is not None:
                await _seed_nova3(store, seed_rate)
            with (
                patch(
                    "coval_bench.pricing.collector._fetch_page",
                    AsyncMock(
                        side_effect=fetch_side_effect,
                        return_value=_fixture_page("deepgram-pricing.html"),
                    ),
                ),
                patch(
                    "coval_bench.pricing.collector._extract_rates",
                    AsyncMock(return_value=rates),
                ),
                patch(
                    "coval_bench.pricing.collector._litellm_prices",
                    AsyncMock(return_value={}),
                ),
                patch("coval_bench.pricing.collector._open_review", _capture_review),
            ):
                summary = await run_update_prices(
                    _SETTINGS, pool, providers=["deepgram"], dry_run=dry_run
                )
            return summary, store
        finally:
            await pool.close()

    summary, store = asyncio.run(_run())
    return summary, reviews, store


def _effective_nova3(pg: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    pg.autocommit = True
    with pg.cursor() as cur:
        cur.execute(
            "SELECT rate_usd, updated_by, superseded_at FROM benchmarks_v2.model_pricing "
            "WHERE provider = 'deepgram' AND model = 'nova-3' ORDER BY id"
        )
        return cur.fetchall()


def test_every_registry_provider_has_a_pricing_source_entry() -> None:
    providers = {m.provider for m in MODEL_REGISTRY}
    assert providers == set(PRICING_SOURCES), (
        "PRICING_SOURCES must have an explicit entry (or None) for every provider"
    )


def test_noop_when_rate_unchanged(pg_conn: psycopg.Connection[Any]) -> None:
    _apply_migrations(pg_conn)
    summary, reviews, _ = _run_collector(pg_conn, [_extracted(0.0048)])
    assert summary["counts"] == {"noop": 1, "auto_applied": 0, "review": 0, "unmatched": 0}
    assert reviews == []
    rows = _effective_nova3(pg_conn)
    assert len(rows) == 1  # nothing new written


def test_small_delta_high_confidence_auto_applies(pg_conn: psycopg.Connection[Any]) -> None:
    _apply_migrations(pg_conn)
    summary, reviews, _ = _run_collector(pg_conn, [_extracted(0.005)])  # +4.2%
    assert summary["counts"]["auto_applied"] == 1
    assert reviews == []
    rows = _effective_nova3(pg_conn)
    assert len(rows) == 2
    old, new = rows
    assert old[2] is not None  # superseded
    assert new[1] == "bot"
    assert new[0] == Decimal("0.00500000")


def test_large_delta_goes_to_review_and_writes_nothing(pg_conn: psycopg.Connection[Any]) -> None:
    _apply_migrations(pg_conn)
    summary, reviews, _ = _run_collector(pg_conn, [_extracted(0.012)])  # +150%
    assert summary["counts"]["review"] == 1
    assert len(reviews) == 1
    title, body = reviews[0]
    assert title == "Price change review: deepgram/nova-3"
    assert "0.012" in body and "0.0048" in body
    assert len(_effective_nova3(pg_conn)) == 1  # untouched


def test_low_confidence_goes_to_review(pg_conn: psycopg.Connection[Any]) -> None:
    _apply_migrations(pg_conn)
    summary, reviews, _ = _run_collector(pg_conn, [_extracted(0.005, confidence="medium")])
    assert summary["counts"]["review"] == 1
    assert len(reviews) == 1
    assert len(_effective_nova3(pg_conn)) == 1


def test_new_model_rate_goes_to_review(pg_conn: psycopg.Connection[Any]) -> None:
    """A rate for a model with no current effective row is never auto-inserted."""
    _apply_migrations(pg_conn)
    summary, reviews, _ = _run_collector(pg_conn, [_extracted(0.0048)], seed_rate=None)
    assert summary["counts"]["review"] == 1
    assert len(reviews) == 1
    assert _effective_nova3(pg_conn) == []


def test_unmatched_model_logged_never_written(pg_conn: psycopg.Connection[Any]) -> None:
    _apply_migrations(pg_conn)
    summary, reviews, _ = _run_collector(pg_conn, [_extracted(0.5, model="Totally Unknown Model")])
    assert summary["counts"]["unmatched"] == 1
    assert reviews == []  # unmatched is a log line, not a review item
    assert len(_effective_nova3(pg_conn)) == 1


def test_fetch_failure_skips_provider_without_aborting(pg_conn: psycopg.Connection[Any]) -> None:
    _apply_migrations(pg_conn)
    summary, reviews, _ = _run_collector(pg_conn, [], fetch_side_effect=RuntimeError("boom"))
    assert summary["failed_providers"] == ["deepgram"]
    # The failure is surfaced through the alert channel.
    assert any("fetch/extract failed" in body for _, body in reviews)


def test_dry_run_writes_nothing(pg_conn: psycopg.Connection[Any]) -> None:
    _apply_migrations(pg_conn)
    summary, reviews, _ = _run_collector(pg_conn, [_extracted(0.005)], dry_run=True)
    assert summary["counts"]["auto_applied"] == 1  # reported in the diff table
    assert reviews == []  # no review items filed on dry-run
    assert len(_effective_nova3(pg_conn)) == 1  # but nothing written

    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM benchmarks_v2.pricing_snapshots")
        row = cur.fetchone()
    assert row is not None and row[0] >= 0  # snapshot table exists either way


def test_snapshot_dedupes_on_hash(pg_conn: psycopg.Connection[Any]) -> None:
    _apply_migrations(pg_conn)

    async def _run() -> None:
        pool = await _make_pool(pg_conn)
        try:
            sha1, changed1 = await _store_snapshot(pool, "deepgram", "https://x", "page-v1")
            sha_same, changed_same = await _store_snapshot(pool, "deepgram", "https://x", "page-v1")
            sha2, changed2 = await _store_snapshot(pool, "deepgram", "https://x", "page-v2")
        finally:
            await pool.close()
        assert sha1 == sha_same and not changed1 and not changed_same
        assert sha2 != sha1 and changed2

    asyncio.run(_run())
    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM benchmarks_v2.pricing_snapshots")
        row = cur.fetchone()
    assert row is not None and row[0] == 2


def test_stale_rates_flagged(pg_conn: psycopg.Connection[Any]) -> None:
    _apply_migrations(pg_conn)

    async def _run() -> list[str]:
        pool = await _make_pool(pg_conn)
        try:
            store = PricingStore(pool)
            await store.upsert_rate(
                provider="deepgram",
                model="nova-3",
                benchmark=Benchmark.STT,
                billing_unit=BillingUnit.PER_MINUTE,
                rate_usd=Decimal("0.0048"),
                source_url="https://deepgram.com/pricing",
                as_of=date.today() - timedelta(days=60),
                updated_by="human",
                effective_at=datetime.now(tz=UTC) - timedelta(days=60),
            )
            return await _stale_rates(store)
        finally:
            await pool.close()

    stale = asyncio.run(_run())
    assert len(stale) == 1
    assert "deepgram/nova-3" in stale[0] and "60d old" in stale[0]


def test_match_model_exact_alias_and_ambiguous() -> None:
    assert _match_model("deepgram", "nova-3") == ("nova-3", Benchmark.STT)
    assert _match_model("deepgram", "Nova-3 Monolingual") == ("nova-3", Benchmark.STT)
    assert _match_model("deepgram", "some-future-model") is None
    # gradium registers 'default' under STT and TTS — ambiguous, never guessed.
    assert _match_model("gradium", "default") is None
