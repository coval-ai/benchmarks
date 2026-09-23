# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for retirement of GET /v1/results."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.api.deps import get_pool, get_settings
from coval_bench.api.internal import hidden_early_access


async def test_legacy_results_returns_410_for_old_filters(client: AsyncClient) -> None:
    response = await client.get(
        "/v1/results",
        params={
            "provider": "deepgram",
            "metric_type": "not-a-legacy-enum",
            "since": "not-a-timestamp",
            "until": "also-not-a-timestamp",
            "include_failed": "not-a-boolean",
            "limit": "not-an-integer",
        },
    )

    assert response.status_code == 410
    assert response.json() == {
        "detail": "The legacy /v1/results endpoint has been retired.",
        "replacement": "/v2/results",
    }
    assert "location" not in response.headers


async def test_legacy_results_does_not_depend_on_database_or_visibility(
    app: FastAPI, client: AsyncClient
) -> None:
    def forbidden() -> None:
        raise AssertionError("retired raw results must not resolve storage or visibility")

    previous = app.dependency_overrides.copy()
    try:
        for dependency in (get_pool, get_settings, hidden_early_access):
            app.dependency_overrides[dependency] = forbidden
        response = await client.get("/v1/results")
        assert response.status_code == 410
    finally:
        app.dependency_overrides = previous


async def test_legacy_results_openapi_does_not_claim_success(client: AsyncClient) -> None:
    document = (await client.get("/openapi.json")).json()
    operation = document["paths"]["/v1/results"]["get"]

    assert "200" not in operation["responses"]
    assert "410" in operation["responses"]
    assert "422" not in operation["responses"]
