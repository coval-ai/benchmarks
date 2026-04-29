# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the slowapi rate limiter.

Note: The in-memory limiter is per-instance. These tests verify the 60 req/min
limit is enforced within a single ASGI app instance. The test resets the limiter
storage between tests to avoid cross-test interference.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from coval_bench.api.ratelimit import limiter


@pytest.fixture(autouse=True)
def reset_limiter() -> None:
    """Reset the in-memory limiter storage before each test."""
    limiter.reset()


async def test_60_requests_all_succeed(app: FastAPI) -> None:
    """60 requests from the same IP all return 200."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        responses = [await c.get("/v1/runs") for _ in range(60)]

    statuses = [r.status_code for r in responses]
    assert all(s == 200 for s in statuses), f"Some requests failed: {statuses}"


async def test_61st_request_returns_429(app: FastAPI) -> None:
    """The 61st request within a minute returns 429 with the expected body."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        for _ in range(60):
            await c.get("/v1/runs")
        response = await c.get("/v1/runs")

    assert response.status_code == 429
    assert response.json() == {"detail": "rate limit exceeded"}


async def test_healthz_exempt_from_ratelimit(app: FastAPI) -> None:
    """GET /healthz is not rate-limited — 100 hits all return 200."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        responses = [await c.get("/healthz") for _ in range(100)]

    assert all(r.status_code == 200 for r in responses)
