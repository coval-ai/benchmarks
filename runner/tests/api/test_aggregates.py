# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GET /v1/results/aggregates."""

from __future__ import annotations

import asyncio
import datetime as dt
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, get_args

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.api.common import WINDOW_INTERVALS, WINDOW_VIEWS, WindowLiteral
from tests.api.conftest import _insert_result, _insert_run, _refresh_mv


def test_intervals_cover_every_window() -> None:
    """Every WindowLiteral value must have an interval and a view — a window
    added to the literal but not the dicts 500s after validation."""
    assert set(WINDOW_INTERVALS) == set(get_args(WindowLiteral))
    assert set(WINDOW_VIEWS) == set(get_args(WindowLiteral))


async def test_empty_db_returns_empty_blocks(client: AsyncClient) -> None:
    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert response.status_code == 200
    body = response.json()
    assert body["benchmark"] == "STT"
    assert body["window"] == "24h"
    assert body["model_stats"] == []
    assert body["series"] == []


async def test_benchmark_required(client: AsyncClient) -> None:
    response = await client.get("/v1/results/aggregates")
    assert response.status_code == 422


async def test_model_stats_math(client: AsyncClient, postgresql: Any) -> None:
    """avg / percentiles / stddev / min / max / count match known values."""
    run_id = await _insert_run(postgresql)
    for value in (1.0, 2.0, 3.0, 4.0):
        await _insert_result(postgresql, run_id, metric_type="WER", metric_value=value)
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert response.status_code == 200
    stats = response.json()["model_stats"]
    assert len(stats) == 1
    s = stats[0]
    assert s["provider"] == "deepgram"
    assert s["model"] == "nova-3"
    assert s["metric_type"] == "WER"
    assert s["avg_value"] == pytest.approx(2.5)
    # percentile_cont linear interpolation
    assert s["p25"] == pytest.approx(1.75)
    assert s["p50"] == pytest.approx(2.5)
    assert s["p75"] == pytest.approx(3.25)
    assert s["p90"] == pytest.approx(3.7)
    assert s["p95"] == pytest.approx(3.85)
    assert s["p99"] == pytest.approx(3.97)
    # sample stddev of 1..4 = sqrt(5/3)
    assert s["stddev_value"] == pytest.approx(1.2909944, rel=1e-6)
    assert s["min_value"] == pytest.approx(1.0)
    assert s["max_value"] == pytest.approx(4.0)
    assert s["sample_count"] == 4


async def test_single_sample_stddev_is_zero(client: AsyncClient, postgresql: Any) -> None:
    """STDDEV_SAMP is NULL for n=1 — must be coalesced to 0 like the client did."""
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, metric_value=3.5)
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    s = response.json()["model_stats"][0]
    assert s["stddev_value"] == 0
    assert s["sample_count"] == 1


async def test_excludes_failed_null_and_other_benchmark(
    client: AsyncClient, postgresql: Any
) -> None:
    """Failed rows, failed parent runs, null metric values, and the other
    benchmark are all excluded from aggregation."""
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, metric_value=1.0)
    # Excluded: failed result row
    await _insert_result(postgresql, run_id, metric_value=100.0, status="failed")
    # Excluded: null metric_value
    await _insert_result(postgresql, run_id, metric_value=None)
    # Excluded: other benchmark
    await _insert_result(
        postgresql, run_id, metric_value=100.0, benchmark="TTS", metric_type="TTFA"
    )
    # Excluded: failed parent run
    failed_run = await _insert_run(postgresql, status="failed")
    await _insert_result(postgresql, failed_run, metric_value=100.0)
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    stats = response.json()["model_stats"]
    assert len(stats) == 1
    assert stats[0]["avg_value"] == pytest.approx(1.0)
    assert stats[0]["sample_count"] == 1


async def test_partial_runs_included(client: AsyncClient, postgresql: Any) -> None:
    run_id = await _insert_run(postgresql, status="partial")
    await _insert_result(postgresql, run_id, metric_value=2.0)
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert response.json()["model_stats"][0]["sample_count"] == 1


async def test_series_buckets_by_scheduled_at(client: AsyncClient, postgresql: Any) -> None:
    """Results from one run share its scheduled_at bucket; values average."""
    scheduled = datetime(2026, 6, 5, 12, 0, 0, tzinfo=dt.UTC)
    run_id = await _insert_run(postgresql, scheduled_at=scheduled)
    await _insert_result(postgresql, run_id, metric_value=1.0)
    await _insert_result(postgresql, run_id, metric_value=3.0)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    series = response.json()["series"]
    assert len(series) == 1
    point = series[0]
    assert datetime.fromisoformat(point["scheduled_at"]) == scheduled
    assert point["avg_value"] == pytest.approx(2.0)
    assert point["sample_count"] == 2


async def test_series_legacy_rows_floor_created_at(client: AsyncClient, postgresql: Any) -> None:
    """Runs without scheduled_at fall back to created_at floored to the
    schedule period (1800s default), same as GET /v1/results."""
    run_id = await _insert_run(postgresql, scheduled_at=None)
    created = datetime.now(dt.UTC) - timedelta(minutes=10)
    await _insert_result(postgresql, run_id, created_at=created, metric_value=1.0)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    series = response.json()["series"]
    assert len(series) == 1
    bucket = datetime.fromisoformat(series[0]["scheduled_at"])
    expected_epoch = created.timestamp() // 1800 * 1800
    assert bucket.timestamp() == pytest.approx(expected_epoch)


async def test_window_excludes_old_rows(client: AsyncClient, postgresql: Any) -> None:
    run_id = await _insert_run(postgresql)
    old = datetime.now(dt.UTC) - timedelta(days=10)
    await _insert_result(postgresql, run_id, created_at=old, metric_value=1.0)
    await _refresh_mv(postgresql)

    response_24h = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert response_24h.json()["model_stats"] == []

    response_30d = await client.get(
        "/v1/results/aggregates", params={"benchmark": "STT", "window": "30d"}
    )
    assert response_30d.json()["model_stats"][0]["sample_count"] == 1


async def test_cache_serves_stale_within_ttl(client: AsyncClient, postgresql: Any) -> None:
    """A second identical request is served from cache, not re-queried."""
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, metric_value=1.0)
    await _refresh_mv(postgresql)

    first = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert first.json()["model_stats"][0]["sample_count"] == 1

    # Out-of-band insert + refresh that a fresh query would pick up.
    await _insert_result(postgresql, run_id, metric_value=3.0)
    await _refresh_mv(postgresql)

    second = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    # Unchanged — served from the cached response, DB not re-scanned.
    assert second.json() == first.json()


async def test_cache_keyed_by_params(client: AsyncClient, postgresql: Any) -> None:
    """Different params are computed independently, not cross-served."""
    run_id = await _insert_run(postgresql)
    old = datetime.now(dt.UTC) - timedelta(days=10)
    await _insert_result(postgresql, run_id, created_at=old, metric_value=1.0)
    await _refresh_mv(postgresql)

    # 24h excludes the 10-day-old row; 30d includes it. Distinct cache keys.
    r_24h = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    r_30d = await client.get("/v1/results/aggregates", params={"benchmark": "STT", "window": "30d"})
    assert r_24h.json()["model_stats"] == []
    assert r_30d.json()["model_stats"][0]["sample_count"] == 1


async def test_concurrent_misses_coalesce(
    client: AsyncClient, app: FastAPI, postgresql: Any
) -> None:
    """Simultaneous uncached requests for one key acquire a pool connection
    once; the rest wait on the per-key lock and read the warmed cache."""
    from coval_bench.api import deps

    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, metric_value=1.0)
    await _refresh_mv(postgresql)

    acquisitions = 0
    real_pool = app.state.pool

    class CountingPool:
        @asynccontextmanager
        async def connection(self) -> Any:
            nonlocal acquisitions
            acquisitions += 1
            # Hold the connection long enough that every gathered request
            # passes the unlocked cache check before the first one fills it.
            await asyncio.sleep(0.2)
            async with real_pool.connection() as conn:
                yield conn

    app.dependency_overrides[deps.get_pool] = CountingPool
    try:
        responses = await asyncio.gather(
            *(client.get("/v1/results/aggregates", params={"benchmark": "STT"}) for _ in range(5))
        )
    finally:
        app.dependency_overrides.clear()

    assert all(r.status_code == 200 for r in responses)
    bodies = [r.json() for r in responses]
    assert all(b == bodies[0] for b in bodies)
    assert bodies[0]["model_stats"][0]["sample_count"] == 1
    assert acquisitions == 1


async def test_failed_fill_shared_not_retried(client: AsyncClient, app: FastAPI) -> None:
    """A failing query is re-raised to requests inside the failure window
    instead of each one re-running it against the pool."""
    from coval_bench.api import deps

    acquisitions = 0

    class FailingPool:
        @asynccontextmanager
        async def connection(self) -> Any:
            nonlocal acquisitions
            acquisitions += 1
            raise RuntimeError("db down")
            yield  # noqa: B901 — unreachable; makes this an async generator

    app.dependency_overrides[deps.get_pool] = FailingPool
    try:
        with pytest.raises(RuntimeError, match="db down"):
            await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
        with pytest.raises(RuntimeError, match="db down"):
            await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    finally:
        app.dependency_overrides.clear()

    assert acquisitions == 1


async def test_models_grouped_separately(client: AsyncClient, postgresql: Any) -> None:
    """Distinct (provider, model, metric_type) groups stay separate and sorted."""
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, provider="deepgram", model="nova-3", metric_value=1.0)
    await _insert_result(postgresql, run_id, provider="assemblyai", model="best", metric_value=2.0)
    await _insert_result(
        postgresql,
        run_id,
        provider="deepgram",
        model="nova-3",
        metric_type="TTFT",
        metric_value=0.5,
    )
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    stats = response.json()["model_stats"]
    keys = [(s["provider"], s["model"], s["metric_type"]) for s in stats]
    assert keys == [
        ("assemblyai", "best", "WER"),
        ("deepgram", "nova-3", "TTFT"),
        ("deepgram", "nova-3", "WER"),
    ]
