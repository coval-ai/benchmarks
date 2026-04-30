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


async def test_response_shape_breaking_change(client: AsyncClient) -> None:
    """models is a list[ModelInfo] (dict with 'model' + 'disabled'), not a list[str]."""
    response = await client.get("/v1/providers")
    data = response.json()
    first_model = data["stt"][0]["models"][0]
    assert isinstance(first_model, dict), "models must be dicts, not strings"
    assert set(first_model.keys()) == {"model", "disabled"}, (
        f"ModelInfo must have exactly 'model' and 'disabled' keys, got {set(first_model.keys())}"
    )

    # Re-activated 2026-04-30 — these must surface in the catalogue with disabled=False.
    openai_entry = next(e for e in data["tts"] if e["provider"] == "openai")
    tts_1_hd = next(m for m in openai_entry["models"] if m["model"] == "tts-1-hd")
    assert tts_1_hd["disabled"] is False

    rime_entry = next(e for e in data["tts"] if e["provider"] == "rime")
    mistv3 = next(m for m in rime_entry["models"] if m["model"] == "mistv3")
    assert mistv3["disabled"] is False


async def test_inactive_tts_models_marked_disabled(client: AsyncClient) -> None:
    """Models the runner doesn't actually run today must report disabled=True.

    Otherwise the FE filter ``!m.disabled`` would let them through and the
    sidebar/legend would render placeholder rows for models we aren't
    benchmarking. Mirrors the cleanup landed alongside Phase 4.7.
    """
    response = await client.get("/v1/providers")
    data = response.json()

    openai_entry = next(e for e in data["tts"] if e["provider"] == "openai")
    for inactive in ("tts-1", "gpt-4o-mini-tts"):
        entry = next(m for m in openai_entry["models"] if m["model"] == inactive)
        assert entry["disabled"] is True, f"{inactive} must be disabled=True"

    rime_entry = next(e for e in data["tts"] if e["provider"] == "rime")
    mistv2 = next(m for m in rime_entry["models"] if m["model"] == "mistv2")
    assert mistv2["disabled"] is True


async def test_providers_no_db_connection(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """The /v1/providers endpoint never acquires a DB connection.

    We verify this by removing the pool from app.state and confirming the
    endpoint still returns 200.
    """
    original_pool = app.state.pool
    app.state.pool = None  # type: ignore[assignment]
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            response = await c.get("/v1/providers")
    finally:
        app.state.pool = original_pool

    assert response.status_code == 200
