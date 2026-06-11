# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GET /v1/leaderboard."""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient

from tests.api.conftest import _insert_result, _insert_run, _refresh_mv


async def test_24h_window_sorted_ascending(client: AsyncClient, postgresql: Any) -> None:
    """window=24h returns entries sorted ascending by avg."""
    run_id = await _insert_run(postgresql)
    # Two providers with different WER values
    await _insert_result(
        postgresql,
        run_id,
        provider="deepgram",
        model="nova-3",
        metric_type="WER",
        metric_value=5.0,
        benchmark="STT",
    )
    await _insert_result(
        postgresql,
        run_id,
        provider="deepgram",
        model="nova-2",
        metric_type="WER",
        metric_value=8.0,
        benchmark="STT",
    )
    await _refresh_mv(postgresql)

    response = await client.get(
        "/v1/leaderboard", params={"metric": "WER", "benchmark": "STT", "window": "24h"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metric"] == "WER"
    assert data["window"] == "24h"
    entries = data["entries"]
    assert len(entries) == 2
    # Ascending by avg
    assert entries[0]["avg"] <= entries[1]["avg"]
    assert entries[0]["model"] == "nova-3"


async def test_incompatible_metric_benchmark_returns_400(client: AsyncClient) -> None:
    """WER + TTS is incompatible — returns 400."""
    response = await client.get("/v1/leaderboard", params={"metric": "WER", "benchmark": "TTS"})
    assert response.status_code == 400


async def test_ttfa_stt_incompatible(client: AsyncClient) -> None:
    """TTFA + STT is incompatible — returns 400."""
    response = await client.get("/v1/leaderboard", params={"metric": "TTFA", "benchmark": "STT"})
    assert response.status_code == 400


async def test_ttft_stt_7d_live_aggregation(client: AsyncClient, postgresql: Any) -> None:
    """window=7d runs live aggregation for TTFT+STT."""
    run_id = await _insert_run(postgresql)
    await _insert_result(
        postgresql,
        run_id,
        provider="deepgram",
        model="nova-3",
        metric_type="TTFT",
        metric_value=120.0,
        metric_units="ms",
        benchmark="STT",
    )

    response = await client.get(
        "/v1/leaderboard",
        params={"metric": "TTFT", "benchmark": "STT", "window": "7d"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metric"] == "TTFT"
    assert data["window"] == "7d"
    assert len(data["entries"]) == 1
    assert data["entries"][0]["provider"] == "deepgram"


async def test_invalid_window_returns_422(client: AsyncClient) -> None:
    """An unknown window value (not 24h/7d/30d) returns 422."""
    response = await client.get(
        "/v1/leaderboard",
        params={"metric": "WER", "benchmark": "STT", "window": "48h"},
    )
    assert response.status_code == 422


async def test_missing_metric_returns_422(client: AsyncClient) -> None:
    """metric is required — omitting it returns 422."""
    response = await client.get("/v1/leaderboard", params={"benchmark": "STT"})
    assert response.status_code == 422


async def test_30d_window_live_aggregation(client: AsyncClient, postgresql: Any) -> None:
    """window=30d runs live aggregation."""
    run_id = await _insert_run(postgresql)
    await _insert_result(
        postgresql,
        run_id,
        provider="deepgram",
        model="nova-2",
        metric_type="WER",
        metric_value=6.0,
        benchmark="STT",
    )
    response = await client.get(
        "/v1/leaderboard",
        params={"metric": "WER", "benchmark": "STT", "window": "30d"},
    )
    assert response.status_code == 200
    assert len(response.json()["entries"]) == 1
