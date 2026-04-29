# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GET /v1/results."""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient

from tests.api.conftest import _insert_result, _insert_run


async def test_filters_compose(client: AsyncClient, postgresql: Any) -> None:
    """Multiple filters AND together — only matching rows returned."""
    run_id = await _insert_run(postgresql)
    # Target row
    await _insert_result(postgresql, run_id, provider="deepgram", model="nova-3", metric_type="WER")
    # Non-matching rows
    await _insert_result(
        postgresql, run_id, provider="openai", model="tts-1", metric_type="TTFA", benchmark="TTS"
    )
    await _insert_result(postgresql, run_id, provider="deepgram", model="nova-2", metric_type="WER")

    response = await client.get(
        "/v1/results",
        params={"provider": "deepgram", "model": "nova-3", "metric": "WER"},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["provider"] == "deepgram"
    assert results[0]["model"] == "nova-3"
    assert results[0]["metric_type"] == "WER"


async def test_since_until_window(client: AsyncClient, postgresql: Any) -> None:
    """Time window narrows results correctly."""
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id)

    # Very old window — should return nothing
    response = await client.get(
        "/v1/results",
        params={
            "since": "2000-01-01T00:00:00Z",
            "until": "2000-12-31T00:00:00Z",
        },
    )
    assert response.status_code == 200
    assert response.json()["results"] == []

    # Broad window — should return the row
    response2 = await client.get(
        "/v1/results",
        params={
            "since": "2020-01-01T00:00:00Z",
            "until": "2099-12-31T00:00:00Z",
        },
    )
    assert response2.status_code == 200
    assert len(response2.json()["results"]) == 1


async def test_since_after_until_returns_400(client: AsyncClient) -> None:
    """since > until returns 400."""
    response = await client.get(
        "/v1/results",
        params={
            "since": "2099-01-01T00:00:00Z",
            "until": "2000-01-01T00:00:00Z",
        },
    )
    assert response.status_code == 400


async def test_failed_results_never_returned(client: AsyncClient, postgresql: Any) -> None:
    """Results with status='failed' are never returned."""
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, status="failed")
    await _insert_result(postgresql, run_id, status="success")

    response = await client.get("/v1/results")
    assert response.status_code == 200
    results = response.json()["results"]
    # Only the success row should appear
    assert len(results) == 1


async def test_empty_db_returns_empty(client: AsyncClient, postgresql: Any) -> None:
    """No results seeded → empty list."""
    response = await client.get("/v1/results")
    assert response.status_code == 200
    assert response.json()["results"] == []
