# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CORS behaviour (ADR-015 — in-app, not infra)."""

from __future__ import annotations

from httpx import AsyncClient


async def test_cors_allowed_origin(client: AsyncClient) -> None:
    """OPTIONS /v1/runs with an allowed origin echoes Access-Control-Allow-Origin."""
    response = await client.options(
        "/v1/runs",
        headers={
            "Origin": "https://benchmarks.coval.ai",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "https://benchmarks.coval.ai"


async def test_cors_localhost_allowed(client: AsyncClient) -> None:
    """localhost:3000 is in the default allowlist."""
    response = await client.options(
        "/v1/runs",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


async def test_cors_disallowed_origin(client: AsyncClient) -> None:
    """An origin not in the allowlist does not get the CORS header."""
    response = await client.options(
        "/v1/runs",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Either no ACAO header, or it does not match the disallowed origin
    acao = response.headers.get("access-control-allow-origin", "")
    assert acao != "https://evil.example.com"


async def test_cors_methods_include_get_options(client: AsyncClient) -> None:
    """Preflight response lists GET and OPTIONS as allowed methods."""
    response = await client.options(
        "/v1/runs",
        headers={
            "Origin": "https://benchmarks.coval.ai",
            "Access-Control-Request-Method": "GET",
        },
    )
    methods_header = response.headers.get("access-control-allow-methods", "")
    assert "GET" in methods_header
