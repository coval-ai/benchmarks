# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GET /v1/providers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


async def test_providers_200(client: AsyncClient) -> None:
    """GET /v1/providers returns 200 with correct shape."""
    response = await client.get("/v1/providers")
    assert response.status_code == 200


async def test_providers_shape(client: AsyncClient) -> None:
    """Response matches ProvidersResponse schema."""
    response = await client.get("/v1/providers")
    data = response.json()
    assert "stt" in data
    assert "tts" in data
    assert isinstance(data["stt"], list)
    assert isinstance(data["tts"], list)


async def test_each_provider_has_models(client: AsyncClient) -> None:
    """Every provider entry has at least one model (ModelInfo dict, not string)."""
    response = await client.get("/v1/providers")
    data = response.json()
    for entry in data["stt"]:
        assert len(entry["models"]) >= 1
        # models are now dicts, not strings
        assert isinstance(entry["models"][0]["model"], str)
    for entry in data["tts"]:
        assert len(entry["models"]) >= 1
        assert isinstance(entry["models"][0]["model"], str)


async def test_disabled_flag_exposed(client: AsyncClient) -> None:
    """Known-disabled models appear with disabled=True; live models appear with disabled=False."""
    response = await client.get("/v1/providers")
    data = response.json()

    # google STT models (chirp_2, long, telephony, short) are all disabled in the matrix
    google_entry = next(e for e in data["stt"] if e["provider"] == "google")
    chirp_2 = next(m for m in google_entry["models"] if m["model"] == "chirp_2")
    assert chirp_2["disabled"] is True

    # deepgram nova-3 is an active model — must be disabled=False
    deepgram_entry = next(e for e in data["stt"] if e["provider"] == "deepgram")
    nova_3 = next(m for m in deepgram_entry["models"] if m["model"] == "nova-3")
    assert nova_3["disabled"] is False

    # xai grok-stt is an active model — must be disabled=False
    xai_entry = next(e for e in data["stt"] if e["provider"] == "xai")
    grok_stt = next(m for m in xai_entry["models"] if m["model"] == "grok-stt")
    assert grok_stt["disabled"] is False


async def test_response_shape_breaking_change(client: AsyncClient) -> None:
    """models is a list[ModelInfo] (dict with 'model', 'disabled', 'tags'), not a list[str]."""
    response = await client.get("/v1/providers")
    data = response.json()
    first_model = data["stt"][0]["models"][0]
    assert isinstance(first_model, dict), "models must be dicts, not strings"
    assert set(first_model.keys()) == {"model", "disabled", "tags"}, (
        f"ModelInfo keys must be model/disabled/tags, got {set(first_model.keys())}"
    )

    openai_entry = next(e for e in data["tts"] if e["provider"] == "openai")
    for active in ("gpt-4o-mini-tts",):
        entry = next(m for m in openai_entry["models"] if m["model"] == active)
        assert entry["disabled"] is False, f"{active} must be disabled=False"

    rime_entry = next(e for e in data["tts"] if e["provider"] == "rime")
    mistv3 = next(m for m in rime_entry["models"] if m["model"] == "mistv3")
    assert mistv3["disabled"] is False


async def test_inactive_tts_models_marked_disabled(client: AsyncClient) -> None:
    """Disabled models must report disabled=True so the FE filter hides them."""
    response = await client.get("/v1/providers")
    data = response.json()

    rime_entry = next(e for e in data["tts"] if e["provider"] == "rime")
    mistv2 = next(m for m in rime_entry["models"] if m["model"] == "mistv2")
    assert mistv2["disabled"] is True

    # OpenAI legacy HTTP models (tts-1, tts-1-hd) are retired but kept disabled.
    openai_entry = next(e for e in data["tts"] if e["provider"] == "openai")
    for legacy in ("tts-1", "tts-1-hd"):
        model = next(m for m in openai_entry["models"] if m["model"] == legacy)
        assert model["disabled"] is True, f"{legacy} must be disabled=True"


async def test_every_model_carries_derived_facets(client: AsyncClient) -> None:
    """Each model emits type/host/lab/source/tenancy facets derived from the registry."""
    response = await client.get("/v1/providers")
    data = response.json()

    xai_entry = next(e for e in data["stt"] if e["provider"] == "xai")
    grok_stt = next(m for m in xai_entry["models"] if m["model"] == "grok-stt")
    facets = {(t["category"], t["value"]) for t in grok_stt["tags"]}
    assert ("type", "STT") in facets
    assert ("host", "xai") in facets
    # lab defaults to the host when no creator override is set; source is then original.
    assert ("lab", "xai") in facets
    assert ("source", "original") in facets
    assert ("tenancy", "shared") in facets

    # groq hosts canopylabs' orpheus, so the creator override drives lab and source.
    groq_entry = next(e for e in data["tts"] if e["provider"] == "groq")
    orpheus = next(m for m in groq_entry["models"] if m["model"] == "canopylabs/orpheus-v1-english")
    orpheus_facets = {(t["category"], t["value"]) for t in orpheus["tags"]}
    assert ("host", "groq") in orpheus_facets
    assert ("lab", "canopylabs") in orpheus_facets
    assert ("source", "inference") in orpheus_facets


async def test_providers_no_db_connection(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """The /v1/providers endpoint never acquires a DB connection.

    We verify this by removing the pool from app.state and confirming the
    endpoint still returns 200.
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

    assert response.status_code == 200
