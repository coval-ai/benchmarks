"""Exercise migrated saved storage through maintenance and real HTTP readers."""

from __future__ import annotations

import datetime as dt
from typing import Any

import psycopg
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.db.dashboard_hourly import refresh_hourly_aggregates
from coval_bench.db.dashboard_summaries import refresh_summary_snapshots
from tests.api.conftest import _insert_run, _make_db_url
from tests.api.test_aggregates import _insert_normalized_bucket, _insert_normalized_metric


@pytest.mark.asyncio
async def test_saved_summary_publication_readiness_cache_and_expiry(
    client: AsyncClient,
    app: FastAPI,
    postgresql: Any,
) -> None:
    app.state.settings.normalized_dashboard_reads_enabled = True
    params = {"benchmark": "STT", "include_series": "false"}
    response = await client.get("/v1/results/aggregates", params=params)
    assert response.status_code == 503
    assert response.json()["detail"] == "dashboard_snapshot_not_ready"
    run = await _insert_run(postgresql)
    await _insert_normalized_metric(
        postgresql, run, dataset_id="stt-v2", metric_type="WER", values={"primary": 10}
    )
    await refresh_summary_snapshots(app.state.pool)
    response = await client.get("/v1/results/aggregates", params=params)
    assert response.status_code == 200, response.text
    first = response.json()
    assert first["model_stats"][0]["avg_value"] == 10
    assert first["datasets"] == ["stt-v2"]
    assert first["snapshot"]["generation"] == 1
    by_dataset = await client.get("/v1/results/aggregates/by-dataset", params={"benchmark": "STT"})
    assert by_dataset.status_code == 200, by_dataset.text
    assert by_dataset.json()["blocks"][0]["model_stats"][0]["avg_value"] == 10
    leaderboard = await client.get("/v1/leaderboard", params={"benchmark": "STT", "metric": "WER"})
    assert leaderboard.status_code == 200, leaderboard.text
    assert leaderboard.json()["snapshot"]["generation"] == 1
    # Reads continue serving generation 1 until the writer publishes generation 2.
    await _insert_normalized_metric(
        postgresql, run, dataset_id="stt-v2", metric_type="WER", values={"primary": 30}
    )
    assert (await client.get("/v1/results/aggregates", params=params)).json() == first
    await refresh_summary_snapshots(app.state.pool)
    updated = (await client.get("/v1/results/aggregates", params=params)).json()
    assert updated["model_stats"][0]["avg_value"] == 20
    assert updated["snapshot"]["generation"] == 2
    # Expiration works even when no ingestion occurs.
    await refresh_summary_snapshots(
        app.state.pool, as_of=dt.datetime.now(dt.UTC) + dt.timedelta(days=31)
    )
    expired = (await client.get("/v1/results/aggregates", params=params)).json()
    assert expired["model_stats"] == [] and expired["datasets"] == []
    assert expired["snapshot"]["generation"] == 3
    async with app.state.pool.connection() as conn:
        await conn.execute(
            "UPDATE benchmarks_v2.dashboard_summary_state SET definition_fingerprint='old'"
        )
    assert (await client.get("/v1/results/aggregates", params=params)).status_code == 503


@pytest.mark.asyncio
async def test_saved_hours_partial_boundaries_and_stale_queue(
    client: AsyncClient,
    app: FastAPI,
    postgresql: Any,
) -> None:
    app.state.settings.normalized_dashboard_reads_enabled = True
    start = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0) - dt.timedelta(
        hours=4
    )
    for minutes, value, count in [
        (0, 999, 1),
        (30, 10, 1),
        (60, 20, 2),
        (90, 60, 3),
        (120, 30, 1),
        (150, 999, 1),
    ]:
        await _insert_normalized_bucket(
            postgresql,
            dataset_id="stt-v2",
            bucket_at=start + dt.timedelta(minutes=minutes),
            value_sum=value,
            sample_count=count,
        )
    params = {
        "benchmark": "STT",
        "dataset": "stt-v2",
        "since": (start + dt.timedelta(minutes=15)).isoformat(),
        "until": (start + dt.timedelta(minutes=135)).isoformat(),
    }
    assert (await client.get("/v1/results/timeline", params=params)).status_code == 503
    # Only the complete middle hour needs a saved state; no summary is required.
    await refresh_hourly_aggregates(app.state.pool, hours=[start + dt.timedelta(hours=1)])
    response = await client.get("/v1/results/timeline", params=params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert [p["value"] for p in body["points"]] == [10, 16, 30]
    assert [p["sample_count"] for p in body["points"]] == [1, 5, 1]
    assert body["materialization"]["stale"] is False
    assert dt.datetime.fromisoformat(body["latest_source_at"]) == start + dt.timedelta(hours=2)
    async with app.state.pool.connection() as conn:
        await conn.execute(
            "INSERT INTO benchmarks_v2.dashboard_source_refreshes(bucket_at) VALUES (%s)",
            (start + dt.timedelta(minutes=30),),
        )
    stale = (await client.get("/v1/results/timeline", params=params)).json()
    assert stale["materialization"]["stale"] is True
    # Both partial boundaries inside one hour must read each source once.
    narrow = await client.get(
        "/v1/results/timeline",
        params={**params, "until": (start + dt.timedelta(minutes=45)).isoformat()},
    )
    assert narrow.status_code == 200, narrow.text
    assert [p["value"] for p in narrow.json()["points"]] == [10]
    assert narrow.json()["materialization"] == {"refreshed_at": None, "stale": True}
    async with app.state.pool.connection() as conn:
        await conn.execute("UPDATE benchmarks_v2.dashboard_hourly_state SET refreshed_at=NULL")
    assert (await client.get("/v1/results/timeline", params=params)).status_code == 503


@pytest.mark.asyncio
async def test_absent_saved_storage_returns_503_but_24h_timeline_still_works(
    client: AsyncClient,
    app: FastAPI,
    postgresql: Any,
) -> None:
    app.state.settings.normalized_dashboard_reads_enabled = True
    with psycopg.connect(_make_db_url(postgresql)) as conn:
        conn.execute("DROP TABLE benchmarks_v2.dashboard_summary_state CASCADE")
    assert (
        await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    ).status_code == 503
    assert (
        await client.get("/v1/results/timeline", params={"benchmark": "STT", "window": "24h"})
    ).status_code == 200
