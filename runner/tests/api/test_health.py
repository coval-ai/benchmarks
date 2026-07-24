# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the health check endpoints."""

from __future__ import annotations

from typing import Any

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


async def test_readyz_closed_pool(app: FastAPI) -> None:
    """GET /readyz with a broken pool returns 503 and leaks no error detail."""

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
    assert response.json() == {"status": "not ready"}
    assert "simulated DB failure" not in response.text


async def test_v1_health_healthy_db(client: AsyncClient) -> None:
    """GET /v1/health with a healthy DB returns 200 with status ok."""
    response = await client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_v1_health_closed_pool(app: FastAPI) -> None:
    """GET /v1/health with a broken pool returns 503 and leaks no error detail."""

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
            response = await c.get("/v1/health")
    finally:
        app.state.pool = original_pool

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
    assert "simulated DB failure" not in response.text
