# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/pricing: the rate in force on a day, its history, and model visibility."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import psycopg
from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.api import deps
from coval_bench.registries import Benchmark, RegisteredModel
from tests.api.conftest import (
    COVAL_ORG,
    EA_MODEL,
    EA_ORG,
    EA_ORG_OTHER,
    EA_PROVIDER,
    SEED_RATES,
    _make_db_url,
    bearer,
)
from tests.roster import TEST_ROSTER

TODAY = datetime.now(UTC).date()
EA_KEY = ("STT", EA_PROVIDER, EA_MODEL)
SEED_KEY = ("STT", "seed", "stt")


def _record(
    postgresql: Any,
    *,
    provider: str = EA_PROVIDER,
    model: str = EA_MODEL,
    price: str | None = "0.006",
    effective_from: date = TODAY,
    recorded_at: datetime | None = None,
) -> None:
    """Append one STT recording straight into the log, as the admin API would."""
    with psycopg.connect(_make_db_url(postgresql)) as conn:
        conn.execute(
            "INSERT INTO benchmarks_v2.pricing_rates (benchmark, provider, model, unit, price_usd,"
            " effective_from, source_url, recorded_by_user_id, recorded_at)"
            " VALUES ('STT', %s, %s, %s, %s::numeric, %s, %s, 'tests', COALESCE(%s, now()))",
            (
                provider,
                model,
                None if price is None else "per_minute",
                price,
                effective_from,
                None if price is None else "https://acme.example.com/pricing",
                recorded_at,
            ),
        )
        conn.commit()


def _serve_model(app: FastAPI, *, collected: bool = True, published: bool = True) -> None:
    """Extend the served roster with the early-access fixture model."""
    model = RegisteredModel(
        benchmark=Benchmark.STT,
        provider=EA_PROVIDER,
        model=EA_MODEL,
        collected=collected,
        published=published,
    )
    app.dependency_overrides[deps.get_models] = lambda: [*TEST_ROSTER, model]


async def _rates(client: AsyncClient, **params: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    response = await client.get("/v1/pricing", params=params)
    assert response.status_code == 200
    return {(r["benchmark"], r["provider"], r["model"]): r for r in response.json()["rates"]}


async def test_the_seeded_log_serves_exact_quotes_with_no_history_yet(client: AsyncClient) -> None:
    response = await client.get("/v1/pricing")
    assert response.json()["as_of"] == TODAY.isoformat()
    rates = await _rates(client)
    assert set(rates) == {(row[0], row[1], row[2]) for row in SEED_RATES}
    stt = rates[SEED_KEY]
    assert (stt["price_usd"], stt["price_per_1k_minutes"], stt["price_per_1m_chars"]) == (
        "0.0048",
        4.8,
        None,
    )
    assert (stt["history"], stt["effective_to"], stt["unit"]) == ([], None, "per_minute")
    assert rates[("TTS", "seed", "tts-a")]["price_per_1m_chars"] == 30.0
    tomorrow = (TODAY + timedelta(days=1)).isoformat()
    assert (await client.get("/v1/pricing", params={"as_of": tomorrow})).status_code == 422


async def test_embargoed_rate_stays_hidden_from_public(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    """A rate on an unpublished model shows only to callers cleared for the model."""
    _serve_model(app, published=False)
    _record(postgresql)
    public = await client.get("/v1/pricing")
    assert EA_KEY not in await _rates(client)
    assert public.headers["X-EA-Token-Status"] == "absent"
    cleared = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG))
    assert cleared.headers["X-EA-Token-Status"] == "accepted"
    assert cleared.headers["Cache-Control"] == "private, no-store"
    rate = next(r for r in cleared.json()["rates"] if r["model"] == EA_MODEL)
    assert (rate["price_per_1k_minutes"], rate["price_per_1m_chars"]) == (6.0, None)
    # A partner org cleared for a different model on the same provider is scoped by model.
    other = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG_OTHER))
    assert all(r["model"] != EA_MODEL for r in other.json()["rates"])
    staff = await client.get("/v1/pricing", headers=bearer(org_id=COVAL_ORG))
    assert any(r["model"] == EA_MODEL for r in staff.json()["rates"])


async def test_the_roster_decides_what_serves(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    """Uncollected models still serve (the site lists them greyed out); unlisted ones never do."""
    _serve_model(app, collected=False)
    _record(postgresql)
    _record(postgresql, provider="ghost", model="nobody")
    rates = await _rates(client)
    assert rates[EA_KEY]["price_per_1k_minutes"] == 6.0
    assert ("STT", "ghost", "nobody") not in rates


async def test_as_of_reads_the_rate_in_force_that_day_with_its_history(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    _serve_model(app)
    day0, day1 = TODAY - timedelta(days=30), TODAY - timedelta(days=10)
    _record(postgresql, price="0.004", effective_from=day0)
    _record(
        postgresql, price="0.005", effective_from=day1, recorded_at=datetime(2026, 9, 1, tzinfo=UTC)
    )
    _record(
        postgresql, price="0.006", effective_from=day1, recorded_at=datetime(2026, 9, 2, tzinfo=UTC)
    )
    _record(
        postgresql, price=None, effective_from=TODAY + timedelta(days=5)
    )  # a scheduled delisting
    _record(postgresql, provider="seed", model="stt", price=None)  # a delisting in force today

    rate = (await _rates(client))[EA_KEY]
    assert (rate["price_usd"], rate["effective_from"]) == ("0.006", day1.isoformat())
    assert rate["effective_to"] is None  # the scheduled change is not public, its date included
    assert [(h["price_usd"], h["effective_to"]) for h in rate["history"]] == [
        ("0.004", day1.isoformat())
    ]

    delisted = (await _rates(client))[SEED_KEY]
    assert (delisted["unit"], delisted["price_usd"], delisted["source_url"]) == (None, None, None)
    assert [(h["price_usd"], h["effective_to"]) for h in delisted["history"]] == [
        ("0.0048", TODAY.isoformat())
    ]

    earlier = (await _rates(client, as_of=day0.isoformat()))[EA_KEY]
    assert (earlier["price_usd"], earlier["history"]) == ("0.004", [])
    assert EA_KEY not in await _rates(client, as_of=(day0 - timedelta(days=1)).isoformat())


async def test_a_missing_log_is_a_503(client: AsyncClient, postgresql: Any) -> None:
    with psycopg.connect(_make_db_url(postgresql)) as conn:
        conn.execute("DROP TABLE benchmarks_v2.pricing_rates")
        conn.commit()
    response = await client.get("/v1/pricing")
    assert response.status_code == 503
    assert response.json()["detail"] == "the pricing log is unavailable"
