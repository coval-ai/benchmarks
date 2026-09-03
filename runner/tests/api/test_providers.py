# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GET /v1/providers.

Each test adds the rows it is about under a made-up provider and reads them
back, so every expectation is written next to the row that produces it.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from coval_bench.registries import Benchmark, Licensing, ModelTag, RegisteredModel, Source
from tests.api.conftest import COVAL_ORG, add_models, bearer


def _model(
    benchmark: Benchmark,
    provider: str,
    model: str,
    *,
    collected: bool = True,
    published: bool = True,
    **fields: Any,
) -> RegisteredModel:
    if benchmark is Benchmark.TTS:
        fields.setdefault("voice", "v")
    return RegisteredModel(
        benchmark=benchmark,
        provider=provider,
        model=model,
        collected=collected,
        published=published,
        **fields,
    )


def _models(payload: dict[str, Any], *, provider: str, **flags: bool) -> set[str]:
    """Models listed for *provider*, filtered by any ModelInfo flags given."""
    return {
        m["model"]
        for board in ("stt", "tts", "s2s", "llm")
        for entry in payload[board]
        if entry["provider"] == provider
        for m in entry["models"]
        if all(m[flag] is value for flag, value in flags.items())
    }


def _entry(payload: dict[str, Any], provider: str, model: str) -> dict[str, Any]:
    return next(
        m
        for board in ("stt", "tts", "s2s", "llm")
        for entry in payload[board]
        if entry["provider"] == provider
        for m in entry["models"]
        if m["model"] == model
    )


def _facets(entry: dict[str, Any]) -> set[tuple[str, str]]:
    return {(t["category"], t["value"]) for t in entry["tags"]}


def _labels(entry: dict[str, Any]) -> dict[tuple[str, str], str]:
    return {(t["category"], t["value"]): t["label"] for t in entry["tags"]}


async def _public(client: AsyncClient) -> dict[str, Any]:
    response = await client.get("/v1/providers")
    assert response.status_code == 200
    return cast("dict[str, Any]", response.json())


async def _internal(client: AsyncClient) -> dict[str, Any]:
    response = await client.get("/v1/providers", headers=bearer(org_id=COVAL_ORG))
    assert response.status_code == 200
    return cast("dict[str, Any]", response.json())


async def test_providers_shape(client: AsyncClient) -> None:
    """Each board is a list of providers; each model is a ModelInfo dict, not a string."""
    data = await _public(client)
    for board in ("stt", "tts", "s2s", "llm"):
        assert isinstance(data[board], list)
    first_model = data["tts"][0]["models"][0]
    assert set(first_model) == {"model", "disabled", "early_access", "tags"}


async def test_publication_alone_decides_public_visibility(
    client: AsyncClient, postgresql: Any
) -> None:
    """`published` is the whole rule: stopping collection never exposes a model."""
    add_models(
        postgresql,
        _model(Benchmark.STT, "acme", "live"),
        _model(Benchmark.STT, "acme", "retired", collected=False),
        _model(Benchmark.STT, "acme", "embargoed", published=False),
        _model(Benchmark.STT, "acme", "shelved", collected=False, published=False),
    )

    public = await _public(client)
    assert _models(public, provider="acme") == {"live", "retired"}
    # Uncollected but published: listed, and marked so a client can grey it out.
    assert _models(public, provider="acme", disabled=True) == {"retired"}


async def test_early_access_flag_marks_only_embargoed_rows(
    client: AsyncClient, postgresql: Any
) -> None:
    """Authorized callers can tell embargoed rows apart; public rows never carry the flag."""
    add_models(
        postgresql,
        _model(Benchmark.STT, "acme", "public"),
        _model(Benchmark.STT, "acme", "embargoed", published=False),
    )

    public = await _public(client)
    assert _models(public, provider="acme") == {"public"}
    assert _models(public, provider="acme", early_access=True) == set()

    internal = await _internal(client)
    assert _models(internal, provider="acme") == {"public", "embargoed"}
    assert _models(internal, provider="acme", early_access=True) == {"embargoed"}
    assert _models(internal, provider="acme", early_access=True, disabled=False) == {"embargoed"}


async def test_a_provider_with_only_embargoed_models_is_absent_from_the_public_view(
    client: AsyncClient, postgresql: Any
) -> None:
    add_models(postgresql, _model(Benchmark.TTS, "acme", "secret", published=False))

    public = await _public(client)
    assert not any(entry["provider"] == "acme" for entry in public["tts"])
    assert _models(await _internal(client), provider="acme") == {"secret"}


async def test_derived_facets_default_to_the_host(client: AsyncClient, postgresql: Any) -> None:
    """Type, host, creator, source, licensing, deployment are derived from the row."""
    add_models(postgresql, _model(Benchmark.STT, "acme", "m"))

    entry = _entry(await _public(client), "acme", "m")
    assert _facets(entry) == {
        ("type", "STT"),
        ("host", "acme"),
        ("creator", "acme"),
        ("source", "official-api"),
        ("licensing", "proprietary"),
        ("deployment", "cloud"),
    }
    # Provider-valued categories keep the raw id, TYPE uppercases, the rest are curated.
    labels = _labels(entry)
    assert labels[("type", "STT")] == "STT"
    assert labels[("host", "acme")] == "acme"
    assert labels[("source", "official-api")] == "Official API"
    assert labels[("licensing", "proprietary")] == "Proprietary"
    assert labels[("deployment", "cloud")] == "Cloud"


async def test_a_creator_override_drives_creator_and_source(
    client: AsyncClient, postgresql: Any
) -> None:
    """A host serving someone else's open weights reports both parties."""
    add_models(
        postgresql,
        _model(
            Benchmark.TTS,
            "acme",
            "other/model",
            creator="other",
            source=Source.SHARED_INFERENCE,
            licensing=Licensing.OPEN_WEIGHT,
        ),
    )

    entry = _entry(await _public(client), "acme", "other/model")
    facets = _facets(entry)
    assert ("host", "acme") in facets
    assert ("creator", "other") in facets
    assert ("source", "shared-inference") in facets
    assert ("licensing", "open-weight") in facets
    assert _labels(entry)[("licensing", "open-weight")] == "Open-weight"


async def test_feature_tags_and_on_prem_surface_as_facets(
    client: AsyncClient, postgresql: Any
) -> None:
    add_models(
        postgresql,
        _model(
            Benchmark.STT,
            "acme",
            "m",
            tags=(ModelTag.DIARIZATION, ModelTag.TRANSLATION),
            on_prem=True,
        ),
    )

    entry = _entry(await _public(client), "acme", "m")
    facets = _facets(entry)
    assert ("features", "diarization") in facets
    assert ("features", "translation") in facets
    assert ("deployment", "on-prem") in facets
    assert _labels(entry)[("features", "diarization")] == "Diarization"


async def test_region_facet_buckets_and_labels(client: AsyncClient, postgresql: Any) -> None:
    """Server location is a coarse us/eu/asia facet with readable labels."""
    add_models(
        postgresql,
        _model(Benchmark.STT, "acme", "us", region="us"),
        _model(Benchmark.STT, "acme", "eu", region="eu"),
        _model(Benchmark.STT, "acme", "asia", region="asia"),
        _model(Benchmark.STT, "acme", "unknown"),
    )

    data = await _public(client)
    regions = {
        model: next(
            (
                (t["value"], t["label"])
                for t in _entry(data, "acme", model)["tags"]
                if t["category"] == "region"
            ),
            None,
        )
        for model in ("us", "eu", "asia", "unknown")
    }
    assert regions == {
        "us": ("us", "US"),
        "eu": ("eu", "Europe"),
        "asia": ("asia", "Asia"),
        "unknown": None,
    }


async def test_tag_categories_metadata(client: AsyncClient) -> None:
    """tag_categories ships the full vocabulary in display order with labels."""
    data = await _public(client)

    categories = data["tag_categories"]
    assert [c["category"] for c in categories] == [
        "type",
        "host",
        "creator",
        "features",
        "source",
        "licensing",
        "deployment",
        "region",
    ]
    by_category = {c["category"]: c for c in categories}
    assert by_category["features"]["label"] == "Features"
    # Host/creator values are provider ids the frontend formats itself.
    assert by_category["host"]["provider_valued"] is True
    assert by_category["creator"]["provider_valued"] is True
    assert by_category["features"]["provider_valued"] is False


async def test_providers_without_a_database_is_503(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalogue comes from the database, so it fails closed without one.

    Serving a stale or invented roster could publish an embargoed model, so
    there is no fallback: the endpoint reports the registry as unavailable.
    """
    original_pool = app.state.pool
    app.state.pool = None
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            response = await c.get("/v1/providers")
    finally:
        app.state.pool = original_pool

    assert response.status_code == 503
