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
from httpx import AsyncClient

from coval_bench.registries import MODEL_REGISTRY, Benchmark, ModelStatus, RegisteredModel
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


def _patch_registry(
    monkeypatch: pytest.MonkeyPatch, entry: PricingEntry, status: ModelStatus
) -> None:
    """Extend the model registry with a model of the given status and price it."""
    patched = [
        *MODEL_REGISTRY,
        RegisteredModel(
            benchmark=entry.benchmark,
            provider=entry.provider,
            model=entry.model,
            status=status,
        ),
    ]
    for module in ("internal", "routers.providers", "routers.pricing"):
        monkeypatch.setattr(f"coval_bench.api.{module}.MODEL_REGISTRY", patched)
    priced = {**PRICING, (entry.benchmark, entry.provider, entry.model): entry}
    monkeypatch.setattr("coval_bench.api.routers.pricing.PRICING", priced)


def _listed_keys() -> set[tuple[str, str, str]]:
    """The packaged rates the public read serves: everything on a listed model."""
    unlisted = {
        (m.benchmark, m.provider, m.model)
        for m in MODEL_REGISTRY
        if m.status in (ModelStatus.RETIRED, ModelStatus.PENDING)
    }
    return {(b.value, p, m) for b, p, m in PRICING if (b, p, m) not in unlisted}


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
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate dated tomorrow is in the registry but prices nothing today."""
    entry = _entry(effective_from=date.today() + timedelta(days=1))
    _patch_registry(monkeypatch, entry, ModelStatus.ACTIVE)
    response = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG))
    assert ("STT", EA_PROVIDER, EA_MODEL) not in _rates_by_key(response.json())


async def test_embargoed_rate_stays_hidden_from_public(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate on an EARLY_ACCESS model shows only to callers cleared for the model."""
    _patch_registry(monkeypatch, _entry(), ModelStatus.EARLY_ACCESS)
    key = ("STT", EA_PROVIDER, EA_MODEL)
    public = await client.get("/v1/pricing")
    assert key not in _rates_by_key(public.json())
    assert public.headers["X-EA-Token-Status"] == "absent"
    cleared = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG))
    assert cleared.headers["X-EA-Token-Status"] == "accepted"
    rate = _rates_by_key(cleared.json())[key]
    assert rate["price_per_1k_minutes"] == 6.0
    assert rate["price_per_1m_chars"] is None


@pytest.mark.parametrize("status", [ModelStatus.RETIRED, ModelStatus.PENDING])
async def test_unlisted_model_rate_serves_to_no_one(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, status: ModelStatus
) -> None:
    """A rate on a retired or pending model is hidden even from cleared callers.

    The site hides these models everywhere; no bearer token re-lists them. The
    ratesheet stays in the repo so the rate returns with the model.
    """
    _patch_registry(monkeypatch, _entry(), status)
    key = ("STT", EA_PROVIDER, EA_MODEL)
    assert key not in _rates_by_key((await client.get("/v1/pricing")).json())
    cleared = await client.get("/v1/pricing", headers=bearer(org_id=EA_ORG))
    assert key not in _rates_by_key(cleared.json())


async def test_no_packaged_rate_prices_an_unlisted_or_unknown_model() -> None:
    """Every shipped ratesheet entry belongs to a model the site currently lists.

    Guards the data, not the route: a rate for a RETIRED or PENDING model would
    silently serve to no one, which is either a stale ratesheet or a status
    change that forgot its pricing. Delist deliberately or re-list the model.
    """
    by_key = {(m.benchmark, m.provider, m.model): m.status for m in MODEL_REGISTRY}
    unlisted = sorted(
        f"{b}:{p}/{m}"
        for (b, p, m) in PRICING
        if by_key[(b, p, m)] in (ModelStatus.RETIRED, ModelStatus.PENDING)
    )
    assert unlisted == [], f"ratesheets price unlisted models: {', '.join(unlisted)}"
