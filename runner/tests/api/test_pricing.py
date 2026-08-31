# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/pricing: packaged rates, manifest usage, and model-visibility rules."""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from importlib.resources import files
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.api import deps
from coval_bench.api.routers import pricing as pricing_router
from coval_bench.registries import MODEL_REGISTRY, Benchmark, RegisteredModel
from coval_bench.registries.pricing import PRICING, PricingEntry
from tests.api.conftest import EA_MODEL, EA_ORG, EA_PROVIDER, bearer


def _entry(**overrides: Any) -> PricingEntry:
    fields: dict[str, Any] = {
        "benchmark": Benchmark.STT,
        "provider": EA_PROVIDER,
        "model": EA_MODEL,
        "unit": "per_minute",
        "price_usd": Decimal("0.006"),
        "effective_from": date.today(),
        "source_url": "https://acme.example.com/pricing",
    }
    fields.update(overrides)
    return PricingEntry(**fields)


def _serve_model(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    entry: PricingEntry,
    *,
    collected: bool = True,
    published: bool = True,
) -> None:
    """Extend the served roster with a model owning *entry* and price it."""
    model = RegisteredModel(
        benchmark=entry.benchmark,
        provider=entry.provider,
        model=entry.model,
        collected=collected,
        published=published,
    )
    app.dependency_overrides[deps.get_models] = lambda: [*MODEL_REGISTRY, model]
    # The router serves the shapes it precomputed at import, so the patch
    # extends those rather than PRICING, which is only read at import time.
    key = (entry.benchmark, entry.provider, entry.model)
    rates = (*pricing_router._RATES, (key, entry.effective_from, pricing_router._rate_out(entry)))
    monkeypatch.setattr(pricing_router, "_RATES", rates)


def _listed_keys() -> set[tuple[str, str, str]]:
    """The packaged rates the public read serves: everything on a published model."""
    published = {(m.benchmark, m.provider, m.model) for m in MODEL_REGISTRY if m.published}
    return {(b.value, p, m) for b, p, m in PRICING if (b, p, m) in published}


def _rates_by_key(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(r["benchmark"], r["provider"], r["model"]): r for r in payload["rates"]}


async def test_get_serves_packaged_rates_and_manifest_usage(client: AsyncClient) -> None:
    """The public read is the packaged registry plus usage recomputed here."""
    response = await client.get("/v1/pricing")
    assert response.status_code == 200
    payload = response.json()
    assert payload["as_of"] == date.today().isoformat()
    keys = set(_rates_by_key(payload))
    assert keys == _listed_keys()

    usage = {u["dataset_id"]: u for u in payload["usage"]}
    manifest = json.loads(
        (files("coval_bench.datasets.manifests") / "tts-v1.json").read_text("utf-8")
    )
    assert usage["tts-v1"]["benchmark"] == "TTS"
    assert usage["tts-v1"]["items"] == len(manifest["items"])
    assert usage["tts-v1"]["characters"] == sum(len(i["transcript"]) for i in manifest["items"])
    assert usage["tts-v1"]["audio_minutes"] is None
    assert usage["stt-v3"]["audio_minutes"] is not None
    assert usage["stt-v3"]["characters"] is None


async def test_price_usd_is_the_exact_decimal_string(client: AsyncClient) -> None:
    """The native figure round-trips verbatim — a float would misquote $0.20 as $0.2."""
    response = await client.get("/v1/pricing")
    served = _rates_by_key(response.json())
    for key in _listed_keys():
        entry = PRICING[(Benchmark(key[0]), key[1], key[2])]
        assert served[key]["price_usd"] == str(entry.price_usd)


async def test_future_rate_is_scheduled_not_current(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate dated tomorrow is in the registry but prices nothing today."""
    entry = _entry(effective_from=date.today() + timedelta(days=1))
    _serve_model(app, monkeypatch, entry)
    response = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG))
    assert ("STT", EA_PROVIDER, EA_MODEL) not in _rates_by_key(response.json())


async def test_embargoed_rate_stays_hidden_from_public(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate on an unpublished model shows only to callers cleared for the model."""
    _serve_model(app, monkeypatch, _entry(), published=False)
    key = ("STT", EA_PROVIDER, EA_MODEL)
    public = await client.get("/v1/pricing")
    assert key not in _rates_by_key(public.json())
    assert public.headers["X-EA-Token-Status"] == "absent"
    cleared = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG))
    assert cleared.headers["X-EA-Token-Status"] == "accepted"
    rate = _rates_by_key(cleared.json())[key]
    assert rate["price_per_1k_minutes"] == 6.0
    assert rate["price_per_1m_chars"] is None


async def test_uncollected_model_rate_serves_to_everyone(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate on a model the runner no longer collects still serves publicly.

    The site lists uncollected models greyed out, so their price stays readable.
    """
    _serve_model(app, monkeypatch, _entry(), collected=False)
    rate = _rates_by_key((await client.get("/v1/pricing")).json())[("STT", EA_PROVIDER, EA_MODEL)]
    assert rate["price_per_1k_minutes"] == 6.0


def test_no_packaged_rate_prices_an_unpublished_model() -> None:
    """Every shipped ratesheet entry belongs to a model the site currently lists.

    Guards the data, not the route: a rate on an unpublished model would
    silently serve to no one, which is either a stale ratesheet or an embargo
    that forgot its pricing. Publish the model or delist deliberately.
    """
    unpublished = sorted(
        f"{m.benchmark}:{m.provider}/{m.model}"
        for m in MODEL_REGISTRY
        if not m.published and (m.benchmark, m.provider, m.model) in PRICING
    )
    assert unpublished == [], f"ratesheets price unpublished models: {', '.join(unpublished)}"
