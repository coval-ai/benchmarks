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
    """Every provider entry has at least one model."""
    response = await client.get("/v1/providers")
    data = response.json()
    for entry in data["stt"]:
        assert len(entry["models"]) >= 1
    for entry in data["tts"]:
        assert len(entry["models"]) >= 1


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
