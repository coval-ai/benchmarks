# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the health check endpoints."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


async def test_healthz_returns_200(client: AsyncClient) -> None:
    """GET /healthz must always return 200 with status ok."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz_healthy_db(client: AsyncClient) -> None:
    """GET /readyz with a healthy DB returns 200 with status ready."""
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


async def test_readyz_closed_pool(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /readyz with a broken pool returns 503 and never raises."""

    # Replace the pool with one that always raises on connection()
    class BrokenPool:
        def connection(self) -> Any:
            raise RuntimeError("simulated DB failure")

    original_pool = app.state.pool
    app.state.pool = BrokenPool()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            response = await c.get("/readyz")
    finally:
        app.state.pool = original_pool

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not ready"
    assert "error" in data
