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

from coval_bench.api.common import (
    MIN_SCORED_SAMPLES,
    WINDOW_INTERVALS,
    WINDOW_VIEWS,
    WindowLiteral,
)
from coval_bench.registries import MODEL_REGISTRY
from tests.api.conftest import _fill_buckets, _insert_result, _insert_run, _refresh_mv


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


async def test_wer_breakdown_averages_and_reconciles(client: AsyncClient, postgresql: Any) -> None:
    """Each error type averages independently, and the three sum to avg_value."""
    run_id = await _insert_run(postgresql)
    for ins, dele, sub in ((1.0, 2.0, 3.0), (3.0, 4.0, 11.0)):
        await _insert_result(
            postgresql,
            run_id,
            metric_value=ins + dele + sub,
            wer_insertions_pct=ins,
            wer_deletions_pct=dele,
            wer_substitutions_pct=sub,
        )
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    s = response.json()["model_stats"][0]
    assert s["wer_insertions_pct"] == pytest.approx(2.0)
    assert s["wer_deletions_pct"] == pytest.approx(3.0)
    assert s["wer_substitutions_pct"] == pytest.approx(7.0)
    assert s["avg_value"] == pytest.approx(12.0)
    parts = ("wer_insertions_pct", "wer_deletions_pct", "wer_substitutions_pct")
    assert sum(s[k] for k in parts) == pytest.approx(s["avg_value"])


async def test_wer_breakdown_null_when_any_row_lacks_it(
    client: AsyncClient, postgresql: Any
) -> None:
    """A scored/pre-migration mix reports no breakdown: a partial average would not reconcile."""
    run_id = await _insert_run(postgresql)
    await _insert_result(
        postgresql,
        run_id,
        metric_value=6.0,
        wer_insertions_pct=1.0,
        wer_deletions_pct=2.0,
        wer_substitutions_pct=3.0,
    )
    await _insert_result(postgresql, run_id, metric_value=10.0)
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    s = response.json()["model_stats"][0]
    assert s["avg_value"] == pytest.approx(8.0)
    assert s["wer_insertions_pct"] is None
    assert s["wer_deletions_pct"] is None
    assert s["wer_substitutions_pct"] is None


async def test_wer_breakdown_null_when_any_component_missing(
    client: AsyncClient, postgresql: Any
) -> None:
    """A row carrying only some components nulls the whole split — two real
    averages beside a null third could never reconcile with avg_value."""
    run_id = await _insert_run(postgresql)
    await _insert_result(
        postgresql,
        run_id,
        metric_value=6.0,
        wer_insertions_pct=1.0,
        wer_deletions_pct=2.0,
    )
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    s = response.json()["model_stats"][0]
    assert s["wer_insertions_pct"] is None
    assert s["wer_deletions_pct"] is None
    assert s["wer_substitutions_pct"] is None


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


async def test_dataset_filter_splits_and_default_pools(
    client: AsyncClient, postgresql: Any
) -> None:
    """Default response pools every dataset; ?dataset= scopes to one; the
    datasets list enumerates what has data."""
    run_v1 = await _insert_run(postgresql, dataset_id="stt-v1")
    await _insert_result(postgresql, run_v1, metric_value=1.0)
    run_v3 = await _insert_run(postgresql, dataset_id="stt-v3")
    await _insert_result(postgresql, run_v3, metric_value=3.0)
    await _insert_result(postgresql, run_v3, metric_value=5.0)
    await _refresh_mv(postgresql)
    await _fill_buckets(postgresql)

    pooled = (await client.get("/v1/results/aggregates", params={"benchmark": "STT"})).json()
    assert pooled["dataset"] == "__all__"
    assert pooled["datasets"] == ["stt-v1", "stt-v3"]
    assert pooled["model_stats"][0]["sample_count"] == 3
    assert pooled["model_stats"][0]["avg_value"] == pytest.approx(3.0)

    scoped = (
        await client.get("/v1/results/aggregates", params={"benchmark": "STT", "dataset": "stt-v3"})
    ).json()
    assert scoped["dataset"] == "stt-v3"
    assert scoped["model_stats"][0]["sample_count"] == 2
    assert scoped["model_stats"][0]["avg_value"] == pytest.approx(4.0)

    missing = (
        await client.get("/v1/results/aggregates", params={"benchmark": "STT", "dataset": "nope"})
    ).json()
    assert missing["model_stats"] == []
    assert missing["datasets"] == ["stt-v1", "stt-v3"]


async def test_tts_rows_attributed_to_tts_dataset(client: AsyncClient, postgresql: Any) -> None:
    """TTS rows aggregate under tts-v1 regardless of the run row's dataset id."""
    run_id = await _insert_run(postgresql, dataset_id="stt-v1")
    await _insert_result(
        postgresql, run_id, metric_value=250.0, benchmark="TTS", metric_type="TTFA"
    )
    await _refresh_mv(postgresql)

    scoped = (
        await client.get("/v1/results/aggregates", params={"benchmark": "TTS", "dataset": "tts-v1"})
    ).json()
    assert scoped["model_stats"][0]["sample_count"] == 1
    assert scoped["datasets"] == ["tts-v1"]


async def test_series_split_by_dataset(client: AsyncClient, postgresql: Any) -> None:
    """The series block scopes to the requested dataset's bucket rows."""
    scheduled = datetime.now(dt.UTC).replace(microsecond=0, second=0, minute=0)
    run_v1 = await _insert_run(postgresql, dataset_id="stt-v1", scheduled_at=scheduled)
    await _insert_result(postgresql, run_v1, metric_value=1.0)
    run_v3 = await _insert_run(postgresql, dataset_id="stt-v3", scheduled_at=scheduled)
    await _insert_result(postgresql, run_v3, metric_value=3.0)
    await _fill_buckets(postgresql)

    pooled = (await client.get("/v1/results/aggregates", params={"benchmark": "STT"})).json()
    assert len(pooled["series"]) == 1
    assert pooled["series"][0]["sample_count"] == 2

    scoped = (
        await client.get("/v1/results/aggregates", params={"benchmark": "STT", "dataset": "stt-v1"})
    ).json()
    assert len(scoped["series"]) == 1
    assert scoped["series"][0]["sample_count"] == 1
    assert scoped["series"][0]["value_sum"] == pytest.approx(1.0)


async def test_partial_runs_included(client: AsyncClient, postgresql: Any) -> None:
    run_id = await _insert_run(postgresql, status="partial")
    await _insert_result(postgresql, run_id, metric_value=2.0)
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert response.json()["model_stats"][0]["sample_count"] == 1


async def test_series_buckets_by_scheduled_at(client: AsyncClient, postgresql: Any) -> None:
    """Results from one run share its scheduled_at bucket; the rollup holds
    the bucket distribution."""
    scheduled = datetime.now(dt.UTC).replace(microsecond=0) - timedelta(hours=1)
    run_id = await _insert_run(postgresql, scheduled_at=scheduled)
    await _insert_result(postgresql, run_id, metric_value=1.0)
    await _insert_result(postgresql, run_id, metric_value=3.0)
    await _fill_buckets(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    series = response.json()["series"]
    assert len(series) == 1
    point = series[0]
    assert datetime.fromisoformat(point["scheduled_at"]) == scheduled
    assert point["min_value"] == pytest.approx(1.0)
    assert point["p25"] == pytest.approx(1.5)
    assert point["p50"] == pytest.approx(2.0)
    assert point["p75"] == pytest.approx(2.5)
    assert point["max_value"] == pytest.approx(3.0)
    assert point["value_sum"] == pytest.approx(4.0)
    assert point["sample_count"] == 2


async def test_series_legacy_rows_floor_created_at(client: AsyncClient, postgresql: Any) -> None:
    """Runs without scheduled_at bucket on created_at floored to the schedule
    period (1800s default)."""
    run_id = await _insert_run(postgresql, scheduled_at=None)
    created = datetime.now(dt.UTC) - timedelta(minutes=10)
    await _insert_result(postgresql, run_id, created_at=created, metric_value=1.0)
    await _fill_buckets(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    series = response.json()["series"]
    assert len(series) == 1
    point = series[0]
    bucket = datetime.fromisoformat(point["scheduled_at"])
    expected_epoch = created.timestamp() // 1800 * 1800
    assert bucket.timestamp() == pytest.approx(expected_epoch)
    assert point["value_sum"] == pytest.approx(1.0)
    assert point["sample_count"] == 1


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
    # The roster shares get_pool; serve it from memory so only the aggregates
    # query is counted here.
    app.dependency_overrides[deps.get_models] = lambda: list(MODEL_REGISTRY)
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
    app.dependency_overrides[deps.get_models] = lambda: list(MODEL_REGISTRY)
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


async def test_by_dataset_empty_db_returns_no_blocks(client: AsyncClient) -> None:
    response = await client.get("/v1/results/aggregates/by-dataset", params={"benchmark": "STT"})
    assert response.status_code == 200
    body = response.json()
    assert body["benchmark"] == "STT"
    assert body["window"] == "24h"
    assert body["blocks"] == []


async def test_by_dataset_groups_stats_per_dataset(client: AsyncClient, postgresql: Any) -> None:
    """One block per dataset, sorted by dataset id; the pooled sentinel rows
    never appear as a block of their own."""
    run_v1 = await _insert_run(postgresql, dataset_id="stt-v1")
    await _insert_result(postgresql, run_v1, metric_value=1.0)
    run_v3 = await _insert_run(postgresql, dataset_id="stt-v3")
    await _insert_result(
        postgresql,
        run_v3,
        metric_value=3.0,
        wer_insertions_pct=0.5,
        wer_deletions_pct=1.0,
        wer_substitutions_pct=1.5,
    )
    await _insert_result(
        postgresql,
        run_v3,
        metric_value=5.0,
        wer_insertions_pct=1.5,
        wer_deletions_pct=2.0,
        wer_substitutions_pct=1.5,
    )
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates/by-dataset", params={"benchmark": "STT"})
    assert response.status_code == 200
    blocks = response.json()["blocks"]
    assert [b["dataset"] for b in blocks] == ["stt-v1", "stt-v3"]

    v1, v3 = blocks
    assert v1["model_stats"][0]["sample_count"] == 1
    assert v1["model_stats"][0]["avg_value"] == pytest.approx(1.0)
    assert v1["model_stats"][0]["wer_insertions_pct"] is None
    assert v3["model_stats"][0]["sample_count"] == 2
    assert v3["model_stats"][0]["avg_value"] == pytest.approx(4.0)
    assert v3["model_stats"][0]["wer_insertions_pct"] == pytest.approx(1.0)
    assert v3["model_stats"][0]["wer_deletions_pct"] == pytest.approx(1.5)
    assert v3["model_stats"][0]["wer_substitutions_pct"] == pytest.approx(1.5)


async def test_by_dataset_respects_window(client: AsyncClient, postgresql: Any) -> None:
    run_id = await _insert_run(postgresql)
    old = datetime.now(dt.UTC) - timedelta(days=10)
    await _insert_result(postgresql, run_id, created_at=old, metric_value=1.0)
    await _refresh_mv(postgresql)

    response_24h = await client.get(
        "/v1/results/aggregates/by-dataset", params={"benchmark": "STT"}
    )
    assert response_24h.json()["blocks"] == []

    response_30d = await client.get(
        "/v1/results/aggregates/by-dataset", params={"benchmark": "STT", "window": "30d"}
    )
    blocks = response_30d.json()["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["model_stats"][0]["sample_count"] == 1


async def test_by_dataset_hides_excluded_metric_rows(client: AsyncClient, postgresql: Any) -> None:
    run_id = await _insert_run(postgresql)
    await _insert_result(
        postgresql,
        run_id,
        provider="assemblyai",
        model="universal-streaming",
        metric_type="TTFS",
        metric_value=0.4,
    )
    await _insert_result(
        postgresql,
        run_id,
        provider="assemblyai",
        model="universal-streaming",
        metric_type="WER",
        metric_value=5.0,
    )
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates/by-dataset", params={"benchmark": "STT"})
    blocks = response.json()["blocks"]
    assert len(blocks) == 1
    metric_types = {s["metric_type"] for s in blocks[0]["model_stats"]}
    assert metric_types == {"WER"}


async def test_by_dataset_cached_separately_from_plain_aggregates(
    client: AsyncClient, postgresql: Any
) -> None:
    """The two endpoints never share a cache entry despite identical params."""
    run_id = await _insert_run(postgresql, dataset_id="stt-v1")
    await _insert_result(postgresql, run_id, metric_value=1.0)
    await _refresh_mv(postgresql)

    plain = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert plain.json()["model_stats"][0]["sample_count"] == 1

    by_dataset = await client.get("/v1/results/aggregates/by-dataset", params={"benchmark": "STT"})
    blocks = by_dataset.json()["blocks"]
    assert [b["dataset"] for b in blocks] == ["stt-v1"]


async def test_excluded_metric_rows_hidden(client: AsyncClient, postgresql: Any) -> None:
    """METRIC_EXCLUSIONS pairs are hidden from stats and series for the excluded
    metric only — other metrics for the same model still show."""
    run_id = await _insert_run(postgresql)
    await _insert_result(
        postgresql,
        run_id,
        provider="assemblyai",
        model="universal-streaming",
        metric_type="TTFS",
        metric_value=0.4,
    )
    await _insert_result(
        postgresql,
        run_id,
        provider="assemblyai",
        model="universal-streaming",
        metric_type="WER",
        metric_value=5.0,
    )
    await _refresh_mv(postgresql)
    await _fill_buckets(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert response.status_code == 200
    body = response.json()
    stat_keys = [(s["provider"], s["model"], s["metric_type"]) for s in body["model_stats"]]
    assert stat_keys == [("assemblyai", "universal-streaming", "WER")]
    series_keys = {(p["provider"], p["model"], p["metric_type"]) for p in body["series"]}
    assert series_keys == {("assemblyai", "universal-streaming", "WER")}


async def test_thin_stat_flagged_but_series_untouched(client: AsyncClient, postgresql: Any) -> None:
    """A thin model stat is flagged; its series points are not.

    One bucket holds a single run's samples, so every point sits under the floor
    by design — flagging them would blank every timeline.
    """
    run_id = await _insert_run(postgresql)
    await _insert_result(
        postgresql,
        run_id,
        provider="hume",
        model="octave-2",
        metric_type="WER",
        metric_value=0.0,
        benchmark="TTS",
    )
    await _refresh_mv(postgresql)
    await _fill_buckets(postgresql)

    response = await client.get(
        "/v1/results/aggregates", params={"benchmark": "TTS", "window": "24h"}
    )
    assert response.status_code == 200
    data = response.json()

    stats = [s for s in data["model_stats"] if s["model"] == "octave-2"]
    assert [s["insufficient_samples"] for s in stats] == [True]
    # The value survives the flag — nothing is dropped or nulled.
    assert stats[0]["sample_count"] == 1
    assert stats[0]["avg_value"] == 0.0

    points = [p for p in data["series"] if p["model"] == "octave-2"]
    assert points, "expected the thin model to still draw a timeline"
    assert all("insufficient_samples" not in p for p in points)


async def test_well_sampled_stat_not_flagged(client: AsyncClient, postgresql: Any) -> None:
    """A model above its modality's floor reports normally."""
    run_id = await _insert_run(postgresql)
    for _ in range(MIN_SCORED_SAMPLES["TTS"] + 1):
        await _insert_result(
            postgresql,
            run_id,
            provider="elevenlabs",
            model="eleven_v3",
            metric_type="WER",
            metric_value=3.0,
            benchmark="TTS",
        )
    await _refresh_mv(postgresql)

    response = await client.get(
        "/v1/results/aggregates", params={"benchmark": "TTS", "window": "24h"}
    )
    stats = [s for s in response.json()["model_stats"] if s["model"] == "eleven_v3"]
    assert [s["insufficient_samples"] for s in stats] == [False]


async def test_by_dataset_stats_carry_the_flag(client: AsyncClient, postgresql: Any) -> None:
    """Per-dataset blocks flag thin stats the same way the pooled ones do.

    The same statistic must not read as trustworthy on one endpoint and thin on
    the other.
    """
    thin_run = await _insert_run(postgresql, dataset_id="stt-v1")
    await _insert_result(postgresql, thin_run, metric_value=0.0)
    full_run = await _insert_run(postgresql, dataset_id="stt-v3")
    for _ in range(MIN_SCORED_SAMPLES["STT"]):
        await _insert_result(postgresql, full_run, metric_value=9.0)
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates/by-dataset", params={"benchmark": "STT"})
    assert response.status_code == 200
    blocks = response.json()["blocks"]
    assert [b["dataset"] for b in blocks] == ["stt-v1", "stt-v3"]

    thin, full = blocks
    assert thin["model_stats"][0]["insufficient_samples"] is True
    # The value survives the flag — nothing is dropped or nulled.
    assert thin["model_stats"][0]["avg_value"] == 0.0
    assert thin["model_stats"][0]["sample_count"] == 1
    assert full["model_stats"][0]["insufficient_samples"] is False
    assert full["model_stats"][0]["sample_count"] == MIN_SCORED_SAMPLES["STT"]


async def test_include_series_false_keeps_stats_and_skips_series_sql(
    client: AsyncClient, postgresql: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lightweight aggregate path must not even prepare a series query."""
    run_id = await _insert_run(postgresql, dataset_id="stt-v1")
    await _insert_result(postgresql, run_id, metric_value=2.0)
    await _refresh_mv(postgresql)

    monkeypatch.setattr(
        "coval_bench.api.routers.aggregates._SERIES_SQL", "SELECT invalid_series_sql"
    )
    monkeypatch.setattr(
        "coval_bench.api.routers.aggregates._COMPACT_SERIES_SQL",
        "SELECT invalid_compact_series_sql",
    )
    response = await client.get(
        "/v1/results/aggregates",
        params={"benchmark": "STT", "window": "30d", "include_series": "false"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["series"] == []
    assert body["datasets"] == ["stt-v1"]
    assert body["model_stats"][0]["avg_value"] == pytest.approx(2.0)


async def test_include_series_cache_variants_do_not_cross_serve(
    client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, metric_value=1.0)
    await _refresh_mv(postgresql)
    await _fill_buckets(postgresql)

    no_series = await client.get(
        "/v1/results/aggregates", params={"benchmark": "STT", "include_series": "false"}
    )
    with_series = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert no_series.json()["series"] == []
    assert len(with_series.json()["series"]) == 1


async def test_timeline_uses_weighted_wer_and_latency_p50(
    client: AsyncClient, postgresql: Any
) -> None:
    scheduled = datetime.now(dt.UTC).replace(microsecond=0)
    run_id = await _insert_run(postgresql, scheduled_at=scheduled)
    await _insert_result(postgresql, run_id, metric_type="WER", metric_value=2.0)
    await _insert_result(postgresql, run_id, metric_type="WER", metric_value=4.0)
    await _insert_result(postgresql, run_id, metric_type="TTFS", metric_value=1.0)
    await _insert_result(postgresql, run_id, metric_type="TTFS", metric_value=9.0)
    await _fill_buckets(postgresql)

    response = await client.get("/v1/results/timeline", params={"benchmark": "STT"})
    assert response.status_code == 200
    values = {point["metric_type"]: point["value"] for point in response.json()["points"]}
    assert values["WER"] == pytest.approx(3.0)
    assert values["TTFS"] == pytest.approx(5.0)


async def test_timeline_short_windows_keep_all_buckets(
    client: AsyncClient, postgresql: Any
) -> None:
    now = datetime.now(dt.UTC).replace(microsecond=0)
    for hours_ago in (1, 24 * 3):
        run_id = await _insert_run(postgresql, scheduled_at=now - timedelta(hours=hours_ago))
        await _insert_result(postgresql, run_id, metric_value=float(hours_ago))
    await _fill_buckets(postgresql)

    seven_days = await client.get(
        "/v1/results/timeline", params={"benchmark": "STT", "window": "7d"}
    )
    assert len(seven_days.json()["points"]) == 2
    one_day = await client.get("/v1/results/timeline", params={"benchmark": "STT", "window": "24h"})
    assert len(one_day.json()["points"]) == 1


async def test_timeline_30d_caps_each_group_and_preserves_endpoints(
    client: AsyncClient, postgresql: Any
) -> None:
    now = datetime.now(dt.UTC).replace(microsecond=0)
    for index in range(241):
        scheduled = now - timedelta(hours=index)
        run_id = await _insert_run(postgresql, scheduled_at=scheduled)
        deepgram_value = -100.0 if index == 80 else 1000.0 if index == 160 else float(index % 37)
        await _insert_result(postgresql, run_id, metric_value=deepgram_value)
        await _insert_result(
            postgresql,
            run_id,
            provider="assemblyai",
            model="best",
            metric_value=float(index),
        )
    await _fill_buckets(postgresql)

    response = await client.get(
        "/v1/results/timeline", params={"benchmark": "STT", "window": "30d"}
    )
    assert response.status_code == 200
    points = response.json()["points"]
    assert points == sorted(
        points,
        key=lambda p: (p["scheduled_at"], p["provider"], p["model"], p["metric_type"]),
    )
    by_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for point in points:
        by_model.setdefault((point["provider"], point["model"]), []).append(point)

    assert set(by_model) == {("assemblyai", "best"), ("deepgram", "nova-3")}
    for group in by_model.values():
        assert len(group) <= 240
        timestamps = {datetime.fromisoformat(point["scheduled_at"]) for point in group}
        assert now in timestamps
        assert now - timedelta(hours=240) in timestamps

    deepgram_values = [point["value"] for point in by_model[("deepgram", "nova-3")]]
    assert min(deepgram_values) == -100
    assert max(deepgram_values) == 1000
    assembly_values = [point["value"] for point in by_model[("assemblyai", "best")]]
    assert min(assembly_values) == 0
    assert max(assembly_values) == 240

    repeated = await client.get(
        "/v1/results/timeline", params={"benchmark": "STT", "window": "30d"}
    )
    assert repeated.json() == response.json()

    legacy = await client.get(
        "/v1/results/aggregates", params={"benchmark": "STT", "window": "30d"}
    )
    legacy_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for point in legacy.json()["series"]:
        legacy_groups.setdefault((point["provider"], point["model"]), []).append(point)
    assert all(len(group) <= 240 for group in legacy_groups.values())
    assert {"min_value", "p25", "p50", "p75", "max_value", "value_sum", "sample_count"} <= set(
        legacy.json()["series"][0]
    )
