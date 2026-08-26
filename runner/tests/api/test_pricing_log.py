# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The pricing_rates change-log: idempotent sync, appended changes, exact scale."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import psycopg
import pytest

from coval_bench.db.pricing_log import sync_pricing_log
from coval_bench.registries.pricing import PRICING
from tests.api.conftest import _make_db_url

# Mirrors migration 20260826_0022, the way this conftest mirrors the matview
# migrations.
_TABLE_SQL = """
    CREATE TABLE benchmarks_v2.pricing_rates (
        id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        benchmark      TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS', 'S2S')),
        provider       TEXT NOT NULL CHECK (provider <> ''),
        model          TEXT NOT NULL CHECK (model <> ''),
        unit           TEXT NOT NULL CHECK (unit <> ''),
        price_usd      NUMERIC NOT NULL CHECK (price_usd > 0),
        effective_from DATE NOT NULL,
        source_url     TEXT NOT NULL CHECK (source_url <> ''),
        notes          TEXT CHECK (notes IS NULL OR notes <> ''),
        recorded_by    TEXT NOT NULL CHECK (recorded_by <> ''),
        recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (benchmark, provider, model, unit, price_usd, effective_from)
    );
    CREATE INDEX pricing_rates_key_recorded_at
        ON benchmarks_v2.pricing_rates (benchmark, provider, model, recorded_at DESC);
"""


@pytest.fixture
def conn(postgresql: Any) -> Iterator[psycopg.Connection]:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as connection:
        connection.execute(_TABLE_SQL)
        yield connection


def test_sync_records_every_packaged_rate_exactly_once(conn: psycopg.Connection) -> None:
    """The first run seeds the log; re-syncing an unchanged registry is a no-op."""
    assert sync_pricing_log(conn) == len(PRICING)
    assert sync_pricing_log(conn) == 0
    row = conn.execute("SELECT count(*) FROM benchmarks_v2.pricing_rates").fetchone()
    assert row is not None and row[0] == len(PRICING)


def test_rate_change_appends_without_rewriting_history(
    conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new price with a new effective date lands one new row; the old row stays."""
    sync_pricing_log(conn)
    key = sorted(PRICING)[0]
    old = PRICING[key]
    changed = old.model_copy(
        update={
            "price_usd": old.price_usd + Decimal("0.001"),
            "effective_from": date.today() + timedelta(days=1),
        }
    )
    monkeypatch.setattr("coval_bench.db.pricing_log.PRICING", {**PRICING, key: changed})
    assert sync_pricing_log(conn) == 1
    rows = conn.execute(
        """
        SELECT price_usd, effective_from FROM benchmarks_v2.pricing_rates
        WHERE benchmark = %s AND provider = %s AND model = %s
        ORDER BY recorded_at, id
        """,
        (key[0].value, key[1], key[2]),
    ).fetchall()
    assert [r[0] for r in rows] == [old.price_usd, changed.price_usd]
    assert rows[0][1] == old.effective_from


def test_price_scale_survives_verbatim(conn: psycopg.Connection) -> None:
    """NUMERIC keeps the printed decimal: $0.20 must never become $0.2."""
    sync_pricing_log(conn)
    trailing_zero = [key for key in PRICING if str(PRICING[key].price_usd).endswith("0")]
    assert trailing_zero, "registry should carry at least one trailing-zero rate"
    for key in trailing_zero:
        row = conn.execute(
            """
            SELECT price_usd::text FROM benchmarks_v2.pricing_rates
            WHERE benchmark = %s AND provider = %s AND model = %s
            """,
            (key[0].value, key[1], key[2]),
        ).fetchone()
        assert row is not None and row[0] == str(PRICING[key].price_usd)
