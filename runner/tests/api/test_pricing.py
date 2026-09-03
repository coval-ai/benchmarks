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
from coval_bench.registries import MODEL_REGISTRY, Benchmark, RegisteredModel
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

TODAY = datetime.now(UTC).date()
EA_KEY = ("STT", EA_PROVIDER, EA_MODEL)


def _record(
    postgresql: Any,
    *,
    benchmark: str = "STT",
    provider: str = EA_PROVIDER,
    model: str = EA_MODEL,
    price: str | None = "0.006",
    unit: str | None = "per_minute",
    effective_from: date = TODAY,
    recorded_at: datetime | None = None,
    source_url: str | None = "https://acme.example.com/pricing",
) -> None:
    """Append one recording straight into the log, as the admin API would."""
    with psycopg.connect(_make_db_url(postgresql)) as conn:
        conn.execute(
            """
            INSERT INTO benchmarks_v2.pricing_rates
                (benchmark, provider, model, unit, price_usd, effective_from, source_url,
                 recorded_by_user_id, recorded_at)
            VALUES (%s, %s, %s, %s, %s::numeric, %s, %s, 'tests', COALESCE(%s, now()))
            """,
            (
                benchmark,
                provider,
                model,
                None if price is None else unit,
                price,
                effective_from,
                None if price is None else source_url,
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
    app.dependency_overrides[deps.get_models] = lambda: [*MODEL_REGISTRY, model]


def _seeded() -> dict[tuple[str, str, str], Any]:
    """The fixture's seeded rates by key: all sit on published models."""
    return {(e.benchmark.value, e.provider, e.model): e for e in SEED_RATES}


def _rates_by_key(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(r["benchmark"], r["provider"], r["model"]): r for r in payload["rates"]}


async def test_get_serves_the_seeded_log(client: AsyncClient) -> None:
    """The public read is the seeded log: each rate in force today, no history yet."""
    response = await client.get("/v1/pricing")
    assert response.status_code == 200
    payload = response.json()
    assert payload["as_of"] == TODAY.isoformat()
    rates = _rates_by_key(payload)
    assert set(rates) == set(_seeded())
    for rate in rates.values():
        assert rate["history"] == []
        assert rate["effective_to"] is None
        assert rate["recorded_at"]


async def test_price_usd_is_the_exact_decimal_string(client: AsyncClient) -> None:
    """The native figure round-trips verbatim — a float would misquote $0.20 as $0.2."""
    served = _rates_by_key((await client.get("/v1/pricing")).json())
    for key, entry in _seeded().items():
        assert served[key]["price_usd"] == str(entry.price_usd)
        assert served[key]["unit"] == str(entry.unit)


async def test_future_rate_is_scheduled_not_current(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    """A rate dated tomorrow is in the log but prices nothing today."""
    _serve_model(app)
    _record(postgresql, effective_from=TODAY + timedelta(days=1))
    response = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG))
    assert EA_KEY not in _rates_by_key(response.json())


async def test_as_of_may_not_be_in_the_future(client: AsyncClient) -> None:
    tomorrow = (TODAY + timedelta(days=1)).isoformat()
    assert (await client.get(f"/v1/pricing?as_of={tomorrow}")).status_code == 422
    assert (await client.get("/v1/pricing?as_of=yesterday")).status_code == 422
    today = await client.get(f"/v1/pricing?as_of={TODAY.isoformat()}")
    assert today.status_code == 200 and today.json()["as_of"] == TODAY.isoformat()


async def test_embargoed_rate_stays_hidden_from_public(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    """A rate on an unpublished model shows only to callers cleared for the model."""
    _serve_model(app, published=False)
    _record(postgresql)
    public = await client.get("/v1/pricing")
    assert EA_KEY not in _rates_by_key(public.json())
    assert public.headers["X-EA-Token-Status"] == "absent"
    cleared = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG))
    assert cleared.headers["X-EA-Token-Status"] == "accepted"
    assert cleared.headers["Cache-Control"] == "private, no-store"
    rate = _rates_by_key(cleared.json())[EA_KEY]
    assert rate["price_per_1k_minutes"] == 6.0
    assert rate["price_per_1m_chars"] is None
    # Another partner org — cleared for a *different* model on the same provider —
    # is scoped by model, not vendor: this rate stays hidden from it.
    other = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG_OTHER))
    assert other.headers["X-EA-Token-Status"] == "accepted"
    assert EA_KEY not in _rates_by_key(other.json())
    # And the coval org sees everything, as it does on every other data endpoint.
    staff = await client.get("/v1/pricing", headers=bearer(org_id=COVAL_ORG))
    assert EA_KEY in _rates_by_key(staff.json())


async def test_uncollected_model_rate_serves_to_everyone(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    """A rate on a model the runner no longer collects still serves publicly.

    The site lists uncollected models greyed out, so their price stays readable.
    """
    _serve_model(app, collected=False)
    _record(postgresql)
    rate = _rates_by_key((await client.get("/v1/pricing")).json())[EA_KEY]
    assert rate["price_per_1k_minutes"] == 6.0


async def test_a_rate_on_a_model_off_the_roster_is_not_served(
    client: AsyncClient, postgresql: Any
) -> None:
    """The log may outlive a model; the read serves only what the roster lists."""
    _record(postgresql, provider="ghost", model="nobody")
    assert ("STT", "ghost", "nobody") not in _rates_by_key((await client.get("/v1/pricing")).json())


async def test_as_of_serves_the_rate_in_force_that_day_with_its_history(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    """A price change mid-timeline: each day reads the rate that held on it."""
    _serve_model(app)
    _record(postgresql, price="0.006", effective_from=date(2026, 1, 1))
    _record(postgresql, price="0.009", effective_from=TODAY)

    now = _rates_by_key((await client.get("/v1/pricing")).json())[EA_KEY]
    assert (now["price_usd"], now["effective_from"], now["effective_to"]) == (
        "0.009",
        TODAY.isoformat(),
        None,
    )
    (old,) = now["history"]
    assert (old["price_usd"], old["effective_from"], old["effective_to"]) == (
        "0.006",
        "2026-01-01",
        TODAY.isoformat(),
    )
    assert old["price_per_1k_minutes"] == 6.0

    yesterday = (TODAY - timedelta(days=1)).isoformat()
    then = _rates_by_key((await client.get(f"/v1/pricing?as_of={yesterday}")).json())[EA_KEY]
    assert (then["price_usd"], then["effective_to"], then["history"]) == (
        "0.006",
        TODAY.isoformat(),
        [],
    )


async def test_a_correction_supersedes_and_stays_out_of_public_history(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    """Re-recording a date replaces the earlier figure; the typo never described a day."""
    _serve_model(app)
    base = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    _record(postgresql, price="0.060", recorded_at=base)
    _record(postgresql, price="0.006", recorded_at=base + timedelta(minutes=5))
    rate = _rates_by_key((await client.get("/v1/pricing")).json())[EA_KEY]
    assert rate["price_usd"] == "0.006"
    assert rate["history"] == []
    assert rate["recorded_at"].startswith("2026-09-01T09:05:00")


async def test_a_delisted_model_serves_null_fields_with_its_history(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    """A model whose provider stopped printing a rate keeps its row and its past."""
    _serve_model(app)
    _record(postgresql, price="0.006", effective_from=date(2026, 1, 1))
    _record(postgresql, price=None, effective_from=TODAY)
    rate = _rates_by_key((await client.get("/v1/pricing")).json())[EA_KEY]
    assert rate["unit"] is None
    assert rate["price_usd"] is None
    assert rate["source_url"] is None
    assert rate["price_per_1k_minutes"] is None
    (old,) = rate["history"]
    assert old["price_usd"] == "0.006" and old["effective_to"] == TODAY.isoformat()
