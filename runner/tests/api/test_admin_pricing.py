# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The admin pricing endpoints over HTTP."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.api import deps
from coval_bench.registries import Benchmark, RegisteredModel
from tests.api.conftest import (
    COVAL_ORG,
    EA_MODEL,
    EA_ORG,
    EA_PROVIDER,
    _make_db_url,
    add_models,
    bearer,
)
from tests.roster import TEST_ROSTER

TODAY = datetime.now(UTC).date()
SEEDED = {"benchmark": "STT", "provider": "seed", "model": "stt"}
ADMIN = bearer(sub="user_admin", org_id=COVAL_ORG, email="admin@coval.dev")


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        **SEEDED,
        "unit": "per_minute",
        "price_usd": "0.0050",
        "source_url": "https://seed.example/pricing",
    }
    return {**body, **overrides}


async def _post(client: AsyncClient, **overrides: Any) -> Any:
    return await client.post("/v1/admin/pricing", json=_body(**overrides), headers=ADMIN)


def _model(payload: dict[str, Any], provider: str = "seed", model: str = "stt") -> dict[str, Any]:
    return next(m for m in payload["models"] if (m["provider"], m["model"]) == (provider, model))


async def test_admin_routes_require_a_coval_token(client: AsyncClient) -> None:
    assert (await client.get("/v1/admin/pricing")).status_code == 401
    assert (
        await client.get("/v1/admin/pricing", headers=bearer(sub="u", org_id=EA_ORG))
    ).status_code == 403
    assert (await client.post("/v1/admin/pricing", json=_body())).status_code == 401


async def test_list_resolves_the_seeded_log_and_is_private(client: AsyncClient) -> None:
    response = await client.get("/v1/admin/pricing", headers=ADMIN)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    model = _model(response.json())
    assert (model["current"]["price_usd"], model["scheduled"]) == ("0.0048", [])
    assert [r["superseded"] for r in model["recordings"]] == [False]


async def test_recording_takes_effect_names_the_admin_and_ignores_a_repeat(
    client: AsyncClient,
) -> None:
    response = await _post(client)
    assert response.status_code == 201
    model = response.json()
    current = model["current"]
    assert (current["price_usd"], current["effective_from"]) == ("0.0050", TODAY.isoformat())
    assert (current["recorded_by_user_id"], current["recorded_by_email"]) == (
        "user_admin",
        "admin@coval.dev",
    )
    newest, seeded = model["recordings"]
    assert newest["price_usd"] == "0.0050" and seeded["recorded_by_user_id"] == "tests"
    assert not newest["superseded"] and not seeded["superseded"]

    served = next(
        r for r in (await client.get("/v1/pricing")).json()["rates"] if r["model"] == "stt"
    )
    assert served["price_usd"] == "0.0050"
    assert served["price_per_1k_minutes"] == current["price_per_1k_minutes"] == 5.0
    assert served["history"][0]["price_usd"] == "0.0048"

    repeat = await _post(client)
    assert repeat.status_code == 200 and len(repeat.json()["recordings"]) == 2


async def test_corrections_supersede_and_future_dates_schedule(client: AsyncClient) -> None:
    await _post(client)
    fix = await _post(client, price_usd="0.0055")
    assert fix.status_code == 201 and fix.json()["current"]["price_usd"] == "0.0055"
    assert [r["superseded"] for r in fix.json()["recordings"]] == [False, True, False]

    ahead = (TODAY + timedelta(days=30)).isoformat()
    scheduled = await _post(
        client, unit=None, price_usd=None, source_url=None, effective_from=ahead
    )
    assert scheduled.status_code == 201
    assert scheduled.json()["current"]["price_usd"] == "0.0055"
    assert [(s["effective_from"], s["price_usd"]) for s in scheduled.json()["scheduled"]] == [
        (ahead, None)
    ]
    served = next(
        r for r in (await client.get("/v1/pricing")).json()["rates"] if r["model"] == "stt"
    )
    assert (served["price_usd"], served["effective_to"]) == ("0.0055", None)


async def test_a_hidden_model_may_be_priced_ahead_of_launch(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    hidden = RegisteredModel(
        benchmark=Benchmark.STT,
        provider=EA_PROVIDER,
        model=EA_MODEL,
        collected=True,
        published=False,
    )
    add_models(postgresql, hidden)
    app.dependency_overrides[deps.get_models] = lambda: [*TEST_ROSTER, hidden]
    response = await _post(
        client,
        provider=EA_PROVIDER,
        model=EA_MODEL,
        price_usd="0.006",
        source_url="https://acme.example.com/p",
    )
    assert response.status_code == 201
    listed = await client.get("/v1/admin/pricing", headers=ADMIN)
    assert _model(listed.json(), EA_PROVIDER, EA_MODEL)["current"]["price_usd"] == "0.006"
    assert all(r["model"] != EA_MODEL for r in (await client.get("/v1/pricing")).json()["rates"])
    cleared = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG))
    assert any(r["model"] == EA_MODEL for r in cleared.json()["rates"])


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({"model": "nova-99"}, "no registered model"),
        ({"price_usd": None}, "both a unit and a price"),
        ({"price_usd": 0.005}, "valid string"),  # a JSON number would drop trailing zeros
        ({"unit": "per_1k_chars"}, "does not bill"),
        ({"source_url": None}, "cite"),
        ({"source_url": "not a url"}, ""),
        ({"price_usd": "0"}, ""),
        ({"effective_from": (TODAY + timedelta(days=400)).isoformat()}, "a year ahead"),
        ({"effective_from": "2019-12-31"}, "may not precede"),
        ({"effective_from": "someday"}, ""),
    ],
)
async def test_writes_are_validated(
    client: AsyncClient, overrides: dict[str, Any], detail: str
) -> None:
    response = await _post(client, **overrides)
    assert response.status_code == 422
    assert detail in str(response.json()["detail"])


async def test_a_missing_log_is_a_503_for_admins_too(client: AsyncClient, postgresql: Any) -> None:
    with psycopg.connect(_make_db_url(postgresql)) as conn:
        conn.execute("DROP TABLE benchmarks_v2.pricing_rates")
        conn.commit()
    assert (await client.get("/v1/admin/pricing", headers=ADMIN)).status_code == 503
    assert (await _post(client)).status_code == 503
