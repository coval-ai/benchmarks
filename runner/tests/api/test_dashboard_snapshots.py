"""Exercise migrated saved storage through maintenance and real HTTP readers."""

from __future__ import annotations

import datetime as dt
from typing import Any

import psycopg
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.db.dashboard_summaries import refresh_summary_snapshots
from tests.api.conftest import _insert_run, _make_db_url
from tests.api.test_aggregates import (
    _insert_normalized_bucket,
    _insert_normalized_metric,
    _publish_timeline_test_hours,
)


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
@pytest.mark.parametrize("age_minutes, stale", [(70, False), (121, True)])
async def test_saved_summary_freshness_allows_hourly_maintenance(
    client: AsyncClient,
    app: FastAPI,
    postgresql: Any,
    age_minutes: int,
    stale: bool,
) -> None:
    app.state.settings.normalized_dashboard_reads_enabled = True
    run = await _insert_run(postgresql)
    await _insert_normalized_metric(
        postgresql, run, dataset_id="stt-v2", metric_type="WER", values={"primary": 10}
    )
    await refresh_summary_snapshots(app.state.pool)
    async with app.state.pool.connection() as conn:
        await conn.execute(
            "UPDATE benchmarks_v2.dashboard_summary_state SET published_at=%s",
            (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=age_minutes),),
        )
    response = await client.get(
        "/v1/results/aggregates", params={"benchmark": "STT", "include_series": "false"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_stats"][0]["avg_value"] == 10
    assert body["snapshot"]["stale"] is stale


@pytest.mark.asyncio
async def test_saved_buckets_serve_only_closed_intervals(
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
    await _publish_timeline_test_hours(postgresql)
    params = {
        "benchmark": "STT",
        "dataset": "stt-v2",
        "since": (start + dt.timedelta(minutes=15)).isoformat(),
        "until": (start + dt.timedelta(minutes=135)).isoformat(),
    }
    response = await client.get("/v1/results/timeline", params=params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert [p["value"] for p in body["points"]] == [16]
    assert [p["sample_count"] for p in body["points"]] == [5]
    assert dt.datetime.fromisoformat(body["latest_source_at"]) == start + dt.timedelta(minutes=90)
    assert "materialization" not in body
    narrow = await client.get(
        "/v1/results/timeline",
        params={**params, "until": (start + dt.timedelta(minutes=105)).isoformat()},
    )
    assert narrow.status_code == 200, narrow.text
    assert narrow.json()["points"] == []


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
