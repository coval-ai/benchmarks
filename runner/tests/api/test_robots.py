# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the robots.txt endpoint."""

from __future__ import annotations

from httpx import AsyncClient


async def test_robots_txt_disallows_all(client: AsyncClient) -> None:
    """GET /robots.txt returns a disallow-all exclusion file."""
    response = await client.get("/robots.txt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "User-agent: *" in response.text
    assert "Disallow: /" in response.text

    openapi = await client.get("/openapi.json")
    assert "/robots.txt" not in openapi.json()["paths"]
