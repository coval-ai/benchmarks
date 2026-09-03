# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The admin pricing endpoints over HTTP."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.api import deps
from coval_bench.registries import MODEL_REGISTRY, Benchmark, RegisteredModel
from coval_bench.registries.pricing import PRICING
from tests.api.conftest import COVAL_ORG, EA_MODEL, EA_ORG, EA_PROVIDER, add_models, bearer

TODAY = datetime.now(UTC).date()
SEEDED = {"benchmark": "STT", "provider": "deepgram", "model": "nova-3"}


def _admin_headers() -> dict[str, str]:
    return bearer(sub="user_admin", org_id=COVAL_ORG, email="admin@coval.dev")


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        **SEEDED,
        "unit": "per_minute",
        "price_usd": "0.0050",
        "source_url": "https://deepgram.com/pricing",
    }
    body.update(overrides)
    return body


async def _post(client: AsyncClient, **overrides: Any) -> Any:
    return await client.post("/v1/admin/pricing", json=_body(**overrides), headers=_admin_headers())


def _model(payload: dict[str, Any], **key: str) -> dict[str, Any]:
    wanted = {**SEEDED, **key}
    return next(
        m
        for m in payload["models"]
        if (m["benchmark"], m["provider"], m["model"])
        == (wanted["benchmark"], wanted["provider"], wanted["model"])
    )


async def test_reads_require_a_coval_token(client: AsyncClient) -> None:
    assert (await client.get("/v1/admin/pricing")).status_code == 401
    partner = bearer(sub="user_partner", org_id=EA_ORG)
    assert (await client.get("/v1/admin/pricing", headers=partner)).status_code == 403
    assert (await client.post("/v1/admin/pricing", json=_body())).status_code == 401
    assert (
        await client.post("/v1/admin/pricing", json=_body(), headers=partner)
    ).status_code == 403


async def test_list_resolves_the_seeded_log_and_is_private(client: AsyncClient) -> None:
    response = await client.get("/v1/admin/pricing", headers=_admin_headers())
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    payload = response.json()
    assert payload["as_of"] == TODAY.isoformat()
    assert len(payload["models"]) == len(PRICING)
    nova = _model(payload)
    seeded = PRICING[(Benchmark.STT, "deepgram", "nova-3")]
    assert nova["current"]["price_usd"] == str(seeded.price_usd)
    assert nova["current"]["price_per_1k_minutes"] == 4.8
    assert nova["scheduled"] == []
    (recording,) = nova["recordings"]
    assert recording["superseded"] is False
    assert recording["recorded_by_user_id"] == "tests"


async def test_recording_a_rate_takes_effect_and_names_the_admin(client: AsyncClient) -> None:
    response = await _post(client)
    assert response.status_code == 201
    model = response.json()
    assert model["current"]["price_usd"] == "0.0050"
    assert model["current"]["effective_from"] == TODAY.isoformat()
    assert model["current"]["recorded_by_email"] == "admin@coval.dev"
    assert model["current"]["recorded_by_user_id"] == "user_admin"
    newest, seeded = model["recordings"]
    assert newest["price_usd"] == "0.0050" and newest["superseded"] is False
    assert seeded["recorded_by_user_id"] == "tests" and seeded["superseded"] is False

    public = await client.get("/v1/pricing")
    served = next(r for r in public.json()["rates"] if r["model"] == "nova-3")
    assert served["price_usd"] == "0.0050"
    # The panel prints the admin figure; the public page prints the served one.
    assert served["price_per_1k_minutes"] == model["current"]["price_per_1k_minutes"] == 5.0
    assert served["history"][0]["price_usd"] == str(
        PRICING[(Benchmark.STT, "deepgram", "nova-3")].price_usd
    )


async def test_repeating_the_current_recording_writes_nothing(client: AsyncClient) -> None:
    assert (await _post(client)).status_code == 201
    repeat = await _post(client)
    assert repeat.status_code == 200
    assert len(repeat.json()["recordings"]) == 2  # the seed plus the one recording


async def test_a_correction_supersedes_the_same_day(client: AsyncClient) -> None:
    await _post(client, price_usd="0.0500")  # slipped a zero
    fixed = await _post(client, price_usd="0.0050")
    assert fixed.status_code == 201
    model = fixed.json()
    assert model["current"]["price_usd"] == "0.0050"
    by_price = {r["price_usd"]: r["superseded"] for r in model["recordings"]}
    assert by_price == {"0.0050": False, "0.0500": True, "0.0048": False}


async def test_a_future_date_schedules_without_changing_today(client: AsyncClient) -> None:
    ahead = (TODAY + timedelta(days=14)).isoformat()
    response = await _post(client, price_usd="0.0077", effective_from=ahead)
    assert response.status_code == 201
    model = response.json()
    assert model["current"]["price_usd"] == "0.0048"
    (scheduled,) = model["scheduled"]
    assert (scheduled["price_usd"], scheduled["effective_from"]) == ("0.0077", ahead)
    public = next(
        r for r in (await client.get("/v1/pricing")).json()["rates"] if r["model"] == "nova-3"
    )
    # The scheduled change is not public until its day — not even its date.
    assert public["price_usd"] == "0.0048" and public["effective_to"] is None


async def test_a_delisting_records_no_public_rate(client: AsyncClient) -> None:
    response = await _post(
        client, unit=None, price_usd=None, source_url=None, notes="No longer on the pricing page."
    )
    assert response.status_code == 201
    current = response.json()["current"]
    assert current["unit"] is None and current["price_usd"] is None
    assert current["notes"] == "No longer on the pricing page."
    public = next(
        r for r in (await client.get("/v1/pricing")).json()["rates"] if r["model"] == "nova-3"
    )
    assert public["price_usd"] is None and public["history"][0]["price_usd"] == "0.0048"


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
    app.dependency_overrides[deps.get_models] = lambda: [*MODEL_REGISTRY, hidden]
    response = await _post(
        client,
        provider=EA_PROVIDER,
        model=EA_MODEL,
        price_usd="0.006",
        source_url="https://acme.example.com/pricing",
    )
    assert response.status_code == 201
    listed = await client.get("/v1/admin/pricing", headers=_admin_headers())
    assert (
        _model(listed.json(), provider=EA_PROVIDER, model=EA_MODEL)["current"]["price_usd"]
        == "0.006"
    )
    public = await client.get("/v1/pricing")
    assert all(r["model"] != EA_MODEL for r in public.json()["rates"])
    cleared = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG))
    assert any(r["model"] == EA_MODEL for r in cleared.json()["rates"])


async def test_effective_from_defaults_to_today(client: AsyncClient) -> None:
    body = _body()
    del body["source_url"]
    body["source_url"] = "https://deepgram.com/pricing"
    response = await client.post("/v1/admin/pricing", json=body, headers=_admin_headers())
    assert response.json()["current"]["effective_from"] == TODAY.isoformat()


async def test_writes_are_validated_like_ratesheet_entries(client: AsyncClient) -> None:
    unknown = await _post(client, model="nova-99")
    assert unknown.status_code == 422 and "no registered model" in unknown.json()["detail"]

    half = await _post(client, price_usd=None)
    assert half.status_code == 422 and "both a unit and a price" in half.json()["detail"]

    assert (await _post(client, price_usd=0.005)).status_code == 422  # a JSON number

    wrong_unit = await _post(client, unit="per_1k_chars")
    assert wrong_unit.status_code == 422 and "does not bill" in wrong_unit.json()["detail"]

    unsourced = await _post(client, source_url=None)
    assert unsourced.status_code == 422 and "cite" in unsourced.json()["detail"]

    assert (await _post(client, source_url="not a url")).status_code == 422
    assert (await _post(client, price_usd="0")).status_code == 422

    too_far = await _post(client, effective_from=(TODAY + timedelta(days=400)).isoformat())
    assert too_far.status_code == 422 and "a year ahead" in too_far.json()["detail"]
    assert (await _post(client, effective_from="2019-12-31")).status_code == 422
    assert (await _post(client, effective_from="someday")).status_code == 422
