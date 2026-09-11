# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GET /v1/results/aggregates."""

from __future__ import annotations

import asyncio
import datetime as dt
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, get_args
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.api.common import (
    MIN_SCORED_SAMPLES,
    WINDOW_INTERVALS,
    WINDOW_VIEWS,
    WindowLiteral,
)
from coval_bench.api.routers.aggregates import (
    _NORMALIZED_DATASETS_SQL,
    _NORMALIZED_SERIES_SQL,
    _NORMALIZED_STATS_BY_DATASET_SQL,
    _NORMALIZED_STATS_SQL,
    _NORMALIZED_TIMELINE_SQL,
    _timeline_bucket_seconds,
)
from coval_bench.registries import TIMELINE_AGGREGATION_RULES, Metric
from coval_bench.registries.metrics import (
    MetricValueContract,
    MetricValueDefinition,
    MetricValueRole,
)
from tests.api.conftest import _fill_buckets, _insert_result, _insert_run, _refresh_mv


async def _insert_normalized_metric(
    postgresql: Any,
    run_id: int,
    *,
    dataset_id: str,
    metric_type: str,
    values: dict[str, float],
    primary_key: str = "primary",
    benchmark: str = "STT",
    observation_status: str = "succeeded",
    evaluation_status: str = "succeeded",
    metric_version: str = "v1",
    evaluation_variant: str = "default",
) -> None:
    """Seed one normalized evaluation for cutover tests."""
    import psycopg

    from tests.api.conftest import _make_db_url

    observation_id, evaluation_id = uuid4(), uuid4()
    async with await psycopg.AsyncConnection.connect(
        _make_db_url(postgresql), autocommit=True
    ) as conn:
        await conn.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (id, run_id, dataset_id, provider, model, benchmark, captured_at, status)
               VALUES (%s, %s, %s, 'deepgram', 'nova-3', %s, now(), %s)""",
            (observation_id, run_id, dataset_id, benchmark, observation_status),
        )
        await conn.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (id, observation_id, metric_type, metric_version, evaluation_variant, status)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                evaluation_id,
                observation_id,
                metric_type,
                metric_version,
                evaluation_variant,
                "running",
            ),
        )
        for key, component in values.items():
            await conn.execute(
                """INSERT INTO benchmarks_v2.metric_values
                   (metric_evaluation_id, value_key, unit, value, value_role)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    evaluation_id,
                    key,
                    "count" if key in _WER_COUNT_KEYS else "percent",
                    component,
                    "primary" if key == primary_key else "component",
                ),
            )
        await conn.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET status = %s WHERE id = %s",
            (evaluation_status, evaluation_id),
        )


_WER_COUNT_KEYS = ("substitution_count", "deletion_count", "insertion_count", "reference_words")


async def _insert_normalized_wer_with_counts(
    postgresql: Any, run_id: int, *, counts: tuple[int, int, int], reference_words: int
) -> None:
    substitutions, deletions, insertions = counts
    await _insert_normalized_metric(
        postgresql,
        run_id,
        dataset_id="stt-v2",
        metric_type="WER",
        values={
            "primary": 100 * sum(counts) / reference_words,
            "substitution_count": substitutions,
            "deletion_count": deletions,
            "insertion_count": insertions,
            "reference_words": reference_words,
        },
    )


async def _insert_normalized_wer(
    postgresql: Any, run_id: int, *, dataset_id: str, value: float
) -> None:
    await _insert_normalized_metric(
        postgresql,
        run_id,
        dataset_id=dataset_id,
        metric_type="WER",
        values={
            "primary": value,
            "insertions": 1.0,
            "deletions": 2.0,
            "substitutions": value - 3.0,
        },
    )


async def _insert_normalized_bucket(
    postgresql: Any,
    *,
    dataset_id: str,
    value_key: str = "primary",
    metric_version: str = "v1",
    value_sum: float = 6.0,
    sample_count: int = 2,
    bucket_at: datetime | None = None,
) -> None:
    import psycopg

    from tests.api.conftest import _make_db_url

    bucket = bucket_at or datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)
    unit = "count" if value_key in _WER_COUNT_KEYS else "percent"
    async with await psycopg.AsyncConnection.connect(
        _make_db_url(postgresql), autocommit=True
    ) as conn:
        await conn.execute(
            """INSERT INTO benchmarks_v2.metric_values_by_bucket
               (provider, model, benchmark, dataset_id, metric_type, metric_version,
                evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50,
                p75, max_value, value_sum, sample_count)
               VALUES ('deepgram', 'nova-3', 'STT', %s, 'WER', %s, 'default', %s,
                       %s, %s, 1, 2, 3, 4, 5, %s, %s)""",
            (dataset_id, metric_version, value_key, unit, bucket, value_sum, sample_count),
        )


def test_intervals_cover_every_window() -> None:
    """Every WindowLiteral value must have an interval and a view — a window
    added to the literal but not the dicts 500s after validation."""
    assert set(WINDOW_INTERVALS) == set(get_args(WindowLiteral))
    assert set(WINDOW_VIEWS) == set(get_args(WindowLiteral))


def test_normalized_query_constants_start_with_sql() -> None:
    """Comments beside triple-quote openers must not become literal SQL."""
    for query in (
        _NORMALIZED_DATASETS_SQL,
        _NORMALIZED_STATS_SQL,
        _NORMALIZED_STATS_BY_DATASET_SQL,
        _NORMALIZED_SERIES_SQL,
        _NORMALIZED_TIMELINE_SQL,
    ):
        assert query.lstrip().startswith(("SELECT", "WITH"))


def test_normalized_dataset_listing_excludes_pooled_sentinel() -> None:
    assert "o.dataset_id <> %(sentinel)s" in _NORMALIZED_DATASETS_SQL


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


async def test_llm_instruction_following_is_served_by_aggregates(
    client: AsyncClient, postgresql: Any
) -> None:
    """The LLM board's pass rate is read here, not from the leaderboard."""
    run_id = await _insert_run(postgresql, dataset_id="llm-dental-v1")
    for value in (100.0, 0.0, 100.0, 100.0):
        await _insert_result(
            postgresql,
            run_id,
            provider="phonely",
            model="phonely-agent",
            metric_type="InstructionFollowing",
            metric_value=value,
            metric_units="percent",
            benchmark="LLM",
        )
    await _refresh_mv(postgresql)

    response = await client.get(
        "/v1/results/aggregates", params={"benchmark": "LLM", "dataset": "llm-dental-v1"}
    )
    assert response.status_code == 200
    stats = response.json()["model_stats"]
    assert [(s["provider"], s["metric_type"], s["sample_count"]) for s in stats] == [
        ("phonely", "InstructionFollowing", 4)
    ]
    assert stats[0]["avg_value"] == pytest.approx(75.0)

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    response = await client.get(
        "/v1/results/aggregates", params={"benchmark": "LLM", "dataset": "llm-dental-v1"}
    )
    assert response.json()["model_stats"] == stats


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


async def test_normalized_dashboard_reads_are_flagged_and_pool_datasets(
    client: AsyncClient, postgresql: Any
) -> None:
    """The default remains legacy; enabling reads uses only normalized rows."""
    legacy_run = await _insert_run(postgresql, dataset_id="legacy")
    await _insert_result(postgresql, legacy_run, metric_value=99.0)
    await _refresh_mv(postgresql)
    normalized_run = await _insert_run(postgresql, dataset_id="stt-v2")
    await _insert_normalized_wer(postgresql, normalized_run, dataset_id="stt-v2", value=6.0)

    # The app fixture builds one Settings instance, so mutating its test-only
    # state exercises the same dependency path as the Cloud Run env flag.
    app = client._transport.app  # type: ignore[attr-defined]
    assert app.state.settings.normalized_dashboard_reads_enabled is False
    legacy = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert legacy.json()["model_stats"][0]["avg_value"] == pytest.approx(99.0)
    app.state.settings.normalized_dashboard_reads_enabled = True

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "__all__"
    assert body["datasets"] == ["stt-v2"]
    assert body["model_stats"][0]["avg_value"] == pytest.approx(6.0)
    assert body["model_stats"][0]["wer_insertions_pct"] == pytest.approx(1.0)

    scoped = await client.get(
        "/v1/results/aggregates", params={"benchmark": "STT", "dataset": "stt-v2"}
    )
    assert scoped.json()["model_stats"][0]["sample_count"] == 1

    by_dataset = await client.get("/v1/results/aggregates/by-dataset", params={"benchmark": "STT"})
    assert [block["dataset"] for block in by_dataset.json()["blocks"]] == ["stt-v2"]

    raw = await client.get("/v1/results", params={"metric_type": "WER"})
    assert [result["metric_value"] for result in raw.json()["results"]] == [99.0]


async def test_normalized_stats_expand_ttfa_and_filter_ineligible_rows(
    client: AsyncClient, postgresql: Any
) -> None:
    valid_run = await _insert_run(postgresql, dataset_id="tts-v1")
    await _insert_normalized_metric(
        postgresql,
        valid_run,
        dataset_id="tts-v1",
        benchmark="TTS",
        metric_type="TTFA",
        values={"primary": 120.0, "roundtrip": 75.0, "leading_silence": 45.0},
    )
    for overrides in (
        {"observation_status": "failed"},
        {"evaluation_status": "failed"},
        {"metric_version": "v2"},
        {"evaluation_variant": "ensemble"},
    ):
        await _insert_normalized_metric(
            postgresql,
            valid_run,
            dataset_id="tts-v1",
            benchmark="TTS",
            metric_type="WER",
            values={"primary": 999.0},
            **overrides,
        )
    failed_run = await _insert_run(postgresql, dataset_id="tts-v1", status="failed")
    await _insert_normalized_metric(
        postgresql,
        failed_run,
        dataset_id="tts-v1",
        benchmark="TTS",
        metric_type="WER",
        values={"primary": 999.0},
    )

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    response = await client.get("/v1/results/aggregates", params={"benchmark": "TTS"})
    stats = {row["metric_type"]: row for row in response.json()["model_stats"]}
    assert set(stats) == {"TTFA", "TTFARoundtrip", "TTFALeadingSilence"}
    assert stats["TTFA"]["avg_value"] == pytest.approx(120.0)
    assert stats["TTFARoundtrip"]["avg_value"] == pytest.approx(75.0)
    assert stats["TTFALeadingSilence"]["avg_value"] == pytest.approx(45.0)


async def test_normalized_component_only_ttfa_is_visible_and_discovers_dataset(
    client: AsyncClient, postgresql: Any
) -> None:
    """TTFA components remain public when the primary role has another key."""
    run_id = await _insert_run(postgresql, dataset_id="tts-components-v1")
    await _insert_normalized_metric(
        postgresql,
        run_id,
        dataset_id="tts-components-v1",
        benchmark="TTS",
        metric_type="TTFA",
        values={"ttfa_ms": 120.0, "roundtrip": 75.0, "leading_silence": 45.0},
        primary_key="ttfa_ms",
    )

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    params = {"benchmark": "TTS", "include_series": "false"}

    pooled = (await client.get("/v1/results/aggregates", params=params)).json()
    pooled_stats = {row["metric_type"]: row for row in pooled["model_stats"]}
    assert set(pooled_stats) == {"TTFARoundtrip", "TTFALeadingSilence"}
    assert pooled["datasets"] == ["tts-components-v1"]

    scoped = (
        await client.get(
            "/v1/results/aggregates",
            params={**params, "dataset": "tts-components-v1"},
        )
    ).json()
    assert {row["metric_type"] for row in scoped["model_stats"]} == {
        "TTFARoundtrip",
        "TTFALeadingSilence",
    }

    by_dataset = (
        await client.get("/v1/results/aggregates/by-dataset", params={"benchmark": "TTS"})
    ).json()["blocks"]
    assert [block["dataset"] for block in by_dataset] == ["tts-components-v1"]
    assert {row["metric_type"] for row in by_dataset[0]["model_stats"]} == {
        "TTFARoundtrip",
        "TTFALeadingSilence",
    }


async def test_normalized_stats_pool_scope_and_group_exact_distribution(
    client: AsyncClient, postgresql: Any
) -> None:
    """Pooled and per dataset normalized stats preserve exact sample math."""
    run_v1 = await _insert_run(postgresql, dataset_id="stt-v1")
    run_v3 = await _insert_run(postgresql, dataset_id="stt-v3")
    for run_id, dataset_id, values in (
        (run_v1, "stt-v1", (1.0, 2.0)),
        (run_v3, "stt-v3", (3.0, 4.0)),
    ):
        for value in values:
            await _insert_normalized_metric(
                postgresql,
                run_id,
                dataset_id=dataset_id,
                metric_type="WER",
                values={"primary": value},
            )

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    params = {"benchmark": "STT", "include_series": "false"}

    pooled = (await client.get("/v1/results/aggregates", params=params)).json()["model_stats"][0]
    assert pooled["sample_count"] == 4
    assert pooled["mean_value"] == pytest.approx(2.5)
    assert pooled["avg_value"] == pytest.approx(2.5)
    assert pooled["stddev_value"] == pytest.approx(1.2909944, rel=1e-6)
    assert pooled["p25"] == pytest.approx(1.75)
    assert pooled["p50"] == pytest.approx(2.5)
    assert pooled["p75"] == pytest.approx(3.25)
    assert pooled["p90"] == pytest.approx(3.7)
    assert pooled["p95"] == pytest.approx(3.85)
    assert pooled["p99"] == pytest.approx(3.97)
    assert pooled["min_value"] == pytest.approx(1.0)
    assert pooled["max_value"] == pytest.approx(4.0)

    scoped = (
        await client.get(
            "/v1/results/aggregates",
            params={**params, "dataset": "stt-v3"},
        )
    ).json()["model_stats"][0]
    assert scoped["sample_count"] == 2
    assert scoped["mean_value"] == pytest.approx(3.5)

    by_dataset = (
        await client.get("/v1/results/aggregates/by-dataset", params={"benchmark": "STT"})
    ).json()["blocks"]
    assert [block["dataset"] for block in by_dataset] == ["stt-v1", "stt-v3"]
    assert [block["model_stats"][0]["mean_value"] for block in by_dataset] == pytest.approx(
        [1.5, 3.5]
    )


async def test_projection_reads_current_metadata_eligibility_and_window(
    client: AsyncClient, postgresql: Any
) -> None:
    """Persisted values never freeze live observation/run attributes or time bounds."""
    import psycopg

    from tests.api.conftest import _make_db_url

    run_id = await _insert_run(postgresql, dataset_id="stt-v2")
    await _insert_normalized_metric(
        postgresql, run_id, dataset_id="stt-v2", metric_type="WER", values={"primary": 10.0}
    )
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True

    async def read(window: str = "24h") -> dict[str, Any]:
        app.state.response_cache.clear()
        response = await client.get(
            "/v1/results/aggregates",
            params={"benchmark": "STT", "include_series": "false", "window": window},
        )
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        return body

    assert (await read())["model_stats"][0]["avg_value"] == pytest.approx(10)
    async with await psycopg.AsyncConnection.connect(
        _make_db_url(postgresql), autocommit=True
    ) as conn:
        await conn.execute(
            "UPDATE benchmarks_v2.benchmark_observations "
            "SET dataset_id = 'stt-renamed', provider = 'Prövïder', model = 'Mödèl' "
            "WHERE run_id = %s",
            (run_id,),
        )
        body = await read()
        assert body["datasets"] == ["stt-renamed"]
        assert body["model_stats"][0]["provider"] == "Prövïder"
        assert body["model_stats"][0]["model"] == "Mödèl"
        await conn.execute(
            "UPDATE benchmarks_v2.runs SET status = 'running' WHERE id = %s", (run_id,)
        )
        body = await read()
        assert body["model_stats"] == []
        assert body["datasets"] == []
        await conn.execute(
            "UPDATE benchmarks_v2.runs SET status = 'partial' WHERE id = %s", (run_id,)
        )
        assert (await read())["model_stats"][0]["avg_value"] == pytest.approx(10)
        await conn.execute(
            "UPDATE benchmarks_v2.benchmark_observations "
            "SET captured_at = now() - interval '2 days' WHERE run_id = %s",
            (run_id,),
        )
        assert (await read())["model_stats"] == []
        assert (await read("7d"))["model_stats"][0]["avg_value"] == pytest.approx(10)
        await conn.execute(
            "UPDATE benchmarks_v2.benchmark_observations SET status = 'failed' WHERE run_id = %s",
            (run_id,),
        )
        body = await read("7d")
        assert body["model_stats"] == []
        assert body["datasets"] == []


async def test_normalized_wer_zero_reference_falls_back_to_mean(
    client: AsyncClient, postgresql: Any
) -> None:
    """Complete count rows with no reference words cannot produce a ratio."""
    run_id = await _insert_run(postgresql, dataset_id="stt-v2")
    await _insert_normalized_metric(
        postgresql,
        run_id,
        dataset_id="stt-v2",
        metric_type="WER",
        values={
            "primary": 0.0,
            "substitution_count": 0.0,
            "deletion_count": 0.0,
            "insertion_count": 0.0,
            "reference_words": 0.0,
        },
    )

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    stat = (
        await client.get(
            "/v1/results/aggregates",
            params={"benchmark": "STT", "include_series": "false"},
        )
    ).json()["model_stats"][0]
    assert stat["sample_count"] == 1
    assert stat["avg_value"] == pytest.approx(0.0)
    assert stat["pooled_value"] is None


async def test_normalized_wer_count_and_split_cohorts_fall_back_independently(
    client: AsyncClient, postgresql: Any
) -> None:
    """Incomplete count or split cohorts cannot manufacture pooled breakdowns."""
    run_id = await _insert_run(postgresql, dataset_id="stt-v2")
    await _insert_normalized_metric(
        postgresql,
        run_id,
        dataset_id="stt-v2",
        metric_type="WER",
        values={
            "primary": 10.0,
            "insertions": 1.0,
            "deletions": 2.0,
            "substitutions": 7.0,
            "substitution_count": 1.0,
            "deletion_count": 1.0,
            "insertion_count": 0.0,
            "reference_words": 10.0,
        },
    )
    await _insert_normalized_metric(
        postgresql,
        run_id,
        dataset_id="stt-v2",
        metric_type="WER",
        values={"primary": 20.0},
    )

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    stat = (
        await client.get(
            "/v1/results/aggregates",
            params={"benchmark": "STT", "include_series": "false"},
        )
    ).json()["model_stats"][0]
    assert stat["sample_count"] == 2
    assert stat["mean_value"] == pytest.approx(15.0)
    assert stat["avg_value"] == pytest.approx(15.0)
    assert stat["pooled_value"] is None
    assert stat["wer_insertions_pct"] is None


async def test_normalized_ttfa_derived_name_collision_keeps_public_count(
    client: AsyncClient, postgresql: Any
) -> None:
    """A native metric sharing a TTFA component name joins the same final group."""
    run_id = await _insert_run(postgresql, dataset_id="tts-v1")
    await _insert_normalized_metric(
        postgresql,
        run_id,
        dataset_id="tts-v1",
        benchmark="TTS",
        metric_type="TTFA",
        values={"primary": 120.0, "roundtrip": 75.0, "leading_silence": 45.0},
    )
    await _insert_normalized_metric(
        postgresql,
        run_id,
        dataset_id="tts-v1",
        benchmark="TTS",
        metric_type="TTFARoundtrip",
        values={"primary": 90.0},
    )

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    stats = {
        row["metric_type"]: row
        for row in (
            await client.get(
                "/v1/results/aggregates",
                params={"benchmark": "TTS", "include_series": "false"},
            )
        ).json()["model_stats"]
    }
    assert stats["TTFARoundtrip"]["sample_count"] == 2
    assert stats["TTFARoundtrip"]["mean_value"] == pytest.approx(82.5)


async def test_normalized_pooled_wer_is_a_ratio_of_sums(
    client: AsyncClient, postgresql: Any
) -> None:
    """One miss on a 4-word clip and a clean 36-word clip: mean 12.5, pooled 2.5."""
    run_id = await _insert_run(postgresql, dataset_id="stt-v2")
    await _insert_normalized_wer_with_counts(
        postgresql, run_id, counts=(1, 0, 0), reference_words=4
    )
    await _insert_normalized_wer_with_counts(
        postgresql, run_id, counts=(0, 0, 0), reference_words=36
    )

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    s = (await client.get("/v1/results/aggregates", params={"benchmark": "STT"})).json()
    s = s["model_stats"][0]
    keys = ("mean_value", "avg_value", "pooled_value", "wer_substitutions_pct")
    assert [s[k] for k in keys] == pytest.approx([12.5, 2.5, 2.5, 2.5])
    assert (s["pooled_deletions_pct"], s["pooled_insertions_pct"]) == (0, 0)
    board = await client.get("/v1/leaderboard", params={"metric": "WER", "benchmark": "STT"})
    assert board.json()["entries"][0]["avg"] == pytest.approx(2.5)

    await _insert_normalized_wer(postgresql, run_id, dataset_id="stt-v2", value=6.0)
    app.state.response_cache.clear()
    s = (await client.get("/v1/results/aggregates", params={"benchmark": "STT"})).json()
    s = s["model_stats"][0]
    assert s["pooled_value"] is None
    assert s["avg_value"] == pytest.approx(s["mean_value"]) == pytest.approx(31 / 3)


async def test_normalized_bucket_pooled_wer_requires_complete_count_rows(
    client: AsyncClient, postgresql: Any
) -> None:
    for key, value_sum in (
        ("substitution_count", 1.0),
        ("deletion_count", 0.0),
        ("insertion_count", 1.0),
        ("reference_words", 40.0),
    ):
        await _insert_normalized_bucket(
            postgresql, dataset_id="__all__", value_key=key, value_sum=value_sum
        )
        await _insert_normalized_bucket(
            postgresql, dataset_id="stt-v2", value_key=key, value_sum=value_sum, sample_count=1
        )
    await _insert_normalized_bucket(postgresql, dataset_id="__all__")
    await _insert_normalized_bucket(postgresql, dataset_id="stt-v2")

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True

    pooled = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    [point] = pooled.json()["series"]
    assert (point["pooled_value"], point["error_sum"], point["reference_word_sum"]) == (5.0, 2, 40)
    timeline = await client.get("/v1/results/timeline", params={"benchmark": "STT"})
    assert timeline.json()["points"][0]["value"] == pytest.approx(5.0)
    compact = await client.get(
        "/v1/results/aggregates", params={"benchmark": "STT", "window": "30d"}
    )
    [point] = compact.json()["series"]
    assert (point["error_sum"], point["reference_word_sum"]) == (2, 40)

    timeline = await client.get(
        "/v1/results/timeline", params={"benchmark": "STT", "dataset": "stt-v2"}
    )
    assert timeline.json()["points"][0]["value"] == pytest.approx(3.0)


async def test_timeline_averages_wer_while_legacy_series_keeps_extrema(
    client: AsyncClient, postgresql: Any
) -> None:
    """Mean WER flat across 360 buckets while pooled spikes once: the spike survives."""
    import psycopg

    from tests.api.conftest import _make_db_url

    now = datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)
    spike_at = now - timedelta(hours=2 * 200)
    rows = []
    for step in range(360):
        bucket = now - timedelta(hours=2 * step)
        reference_words = 10.0 if bucket == spike_at else 90.0
        for key, value_sum in (
            ("primary", 20.0),
            ("substitution_count", 9.0),
            ("deletion_count", 0.0),
            ("insertion_count", 0.0),
            ("reference_words", reference_words),
        ):
            rows.append((key, "count" if key in _WER_COUNT_KEYS else "percent", bucket, value_sum))
    async with (
        await psycopg.AsyncConnection.connect(_make_db_url(postgresql), autocommit=True) as conn,
        conn.cursor() as cur,
    ):
        await cur.executemany(
            """INSERT INTO benchmarks_v2.metric_values_by_bucket
                   (provider, model, benchmark, dataset_id, metric_type, metric_version,
                    evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50,
                    p75, max_value, value_sum, sample_count)
                   VALUES ('deepgram', 'nova-3', 'STT', '__all__', 'WER', 'v1', 'default',
                           %s, %s, %s, 1, 2, 3, 4, 5, %s, 2)""",
            rows,
        )

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    response = await client.get(
        "/v1/results/timeline", params={"benchmark": "STT", "window": "30d"}
    )
    points = response.json()["points"]
    assert len(points) < 360
    assert max(p["value"] for p in points) == pytest.approx(18.0)
    legacy = await client.get(
        "/v1/results/aggregates", params={"benchmark": "STT", "window": "30d"}
    )
    assert max(p["pooled_value"] for p in legacy.json()["series"]) == pytest.approx(90.0)


async def test_normalized_series_and_timeline_use_primary_v1_default_buckets(
    client: AsyncClient, postgresql: Any
) -> None:
    await _insert_normalized_bucket(postgresql, dataset_id="__all__")
    await _insert_normalized_bucket(postgresql, dataset_id="stt-v2", value_sum=4.0, sample_count=1)
    await _insert_normalized_bucket(postgresql, dataset_id="__all__", value_key="insertions")
    await _insert_normalized_bucket(postgresql, dataset_id="__all__", metric_version="v2")

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True

    scoped = await client.get(
        "/v1/results/aggregates", params={"benchmark": "STT", "dataset": "stt-v2"}
    )
    assert len(scoped.json()["series"]) == 1
    assert scoped.json()["series"][0]["value_sum"] == pytest.approx(4.0)

    timeline = await client.get("/v1/results/timeline", params={"benchmark": "STT"})
    assert len(timeline.json()["points"]) == 1
    assert timeline.json()["points"][0]["value"] == pytest.approx(3.0)


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
    app.dependency_overrides[deps.get_models] = lambda: []
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
    app.dependency_overrides[deps.get_models] = lambda: []
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


def test_timeline_bucket_chooser_is_smallest_supported_interval() -> None:
    """Adaptive ranges stay at or below roughly 200 points without tiny buckets."""
    assert _timeline_bucket_seconds(1) == 3600
    assert _timeline_bucket_seconds(200 * 3600) == 3600
    assert _timeline_bucket_seconds(201 * 3600) == 7200
    assert _timeline_bucket_seconds(200 * 86400) == 86400


async def test_timeline_rejects_invalid_custom_bounds(client: AsyncClient) -> None:
    cases = (
        {"since": "2026-09-01T00:00:00", "until": "2026-09-01T01:00:00+00:00"},
        {"since": "2026-09-01T01:00:00+00:00"},
        {"since": "2026-09-01T01:00:00+00:00", "until": "2026-09-01T01:00:00+00:00"},
        {
            "since": "2026-09-02T00:00:00+00:00",
            "until": "2026-09-01T00:00:00+00:00",
        },
        {
            "since": "2026-08-01T00:00:00+00:00",
            "until": "2026-09-01T00:00:00+00:00",
        },
        {
            "window": "7d",
            "since": "2026-09-01T00:00:00+00:00",
            "until": "2026-09-01T01:00:00+00:00",
        },
    )
    for bounds in cases:
        response = await client.get("/v1/results/timeline", params={"benchmark": "STT", **bounds})
        assert response.status_code == 422, (bounds, response.text)


async def test_timeline_explicit_24h_preserves_run_aggregation_metadata(
    client: AsyncClient, postgresql: Any
) -> None:
    run_id = await _insert_run(postgresql, scheduled_at=datetime.now(dt.UTC))
    await _insert_result(postgresql, run_id, metric_value=3.0)
    await _fill_buckets(postgresql)

    response = await client.get(
        "/v1/results/timeline", params={"benchmark": "STT", "window": "24h"}
    )
    body = response.json()
    assert body["window"] == "24h"
    assert body["aggregation"] == "run"
    assert body["bucket_seconds"] is None
    assert all(
        p["aggregation_method"] is None and p["sample_count"] is None for p in body["points"]
    )
    assert body["latest_source_at"] is not None


@pytest.mark.parametrize("normalized", [False, True])
async def test_timeline_historical_bounds_groups_cache_and_visibility(
    client: AsyncClient,
    postgresql: Any,
    app: FastAPI,
    normalized: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact historical bounds, partial bins, and hidden freshness survive aggregation."""
    import psycopg

    from coval_bench.api.internal import hidden_early_access
    from tests.api.conftest import _make_db_url

    monkeypatch.setitem(
        TIMELINE_AGGREGATION_RULES, "FutureMetric", TIMELINE_AGGREGATION_RULES["TTFS"]
    )
    app.state.settings.normalized_dashboard_reads_enabled = normalized
    start = datetime(2026, 1, 1, 0, 30, tzinfo=dt.UTC)
    end = start + timedelta(days=7)
    rows = [
        ("nova-3", "TTFS", "__all__", start - timedelta(seconds=1), 1000, 1),
        ("nova-3", "TTFS", "__all__", start, 10, 1),
        ("nova-3", "TTFS", "__all__", start + timedelta(minutes=15), 90, 3),
        ("nova-3", "TTFS", "__all__", end, 500, 1),
        ("nova-3", "FutureMetric", "__all__", start, 80, 2),
        ("nova-3", "TTFS", "other-dataset", start, 60, 2),
        ("hidden-model", "TTFS", "__all__", end - timedelta(seconds=1), 5, 1),
    ]
    async with await psycopg.AsyncConnection.connect(
        _make_db_url(postgresql), autocommit=True
    ) as conn:
        for model, metric, dataset, source_at, total, count in rows:
            source_params = (model, metric, dataset, source_at, total, count)
            if normalized:
                await conn.execute(
                    """INSERT INTO benchmarks_v2.metric_values_by_bucket
                    (provider,model,metric_type,dataset_id,bucket_at,value_sum,sample_count,
                     benchmark,metric_version,evaluation_variant,value_key,unit,min_value,p25,p50,p75,max_value)
                    VALUES ('deepgram',%s,%s,%s,%s,%s,%s,'STT','v1','default',
                            'primary','seconds',1,2,999,1000,1001)""",
                    source_params,
                )
            else:
                await conn.execute(
                    """INSERT INTO benchmarks_v2.results_by_bucket
                    (provider,model,metric_type,dataset_id,bucket_at,value_sum,sample_count,
                     benchmark,min_value,p25,p50,p75,max_value)
                    VALUES ('deepgram',%s,%s,%s,%s,%s,%s,'STT',1,2,999,1000,1001)""",
                    source_params,
                )
    params = {"benchmark": "STT", "since": start.isoformat(), "until": end.isoformat()}
    app.dependency_overrides[hidden_early_access] = lambda: frozenset(
        {("deepgram", "hidden-model")}
    )
    response = await client.get("/v1/results/timeline", params=params)
    assert response.status_code == 200
    body = response.json()
    assert body["bucket_seconds"] == 3600
    assert all(p["aggregation_method"] == "mean" for p in body["points"])
    assert {p["metric_type"]: p["value"] for p in body["points"]} == {
        "TTFS": 25,
        "FutureMetric": 40,
    }
    assert all(
        datetime.fromisoformat(p["scheduled_at"]) == start.replace(minute=0) for p in body["points"]
    )
    assert datetime.fromisoformat(body["latest_source_at"]) == start + timedelta(minutes=15)
    # Equivalent offsets reuse the same bounds; distinct bounds must not reuse rows.
    offset = dt.timezone(timedelta(hours=-7))
    same = await client.get(
        "/v1/results/timeline",
        params={
            **params,
            "since": start.astimezone(offset).isoformat(),
            "until": end.astimezone(offset).isoformat(),
        },
    )
    assert same.json() == body
    later = await client.get(
        "/v1/results/timeline",
        params={
            **params,
            "since": end.isoformat(),
            "until": (end + timedelta(hours=1)).isoformat(),
        },
    )
    assert [p["value"] for p in later.json()["points"]] == [500]
    scoped_response = await client.get(
        "/v1/results/timeline", params={**params, "dataset": "other-dataset"}
    )
    assert [p["value"] for p in scoped_response.json()["points"]] == [30]
    app.dependency_overrides[hidden_early_access] = lambda: frozenset()
    visible = await client.get("/v1/results/timeline", params=params)
    assert len(visible.json()["points"]) == 3
    assert datetime.fromisoformat(visible.json()["latest_source_at"]) == end - timedelta(seconds=1)


async def test_timeline_custom_average_weights_source_counts(
    client: AsyncClient, postgresql: Any
) -> None:
    """Averages use pooled sample counts instead of medians or mean-of-means."""
    start = datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    first = await _insert_run(postgresql, scheduled_at=start)
    await _insert_result(postgresql, first, metric_value=1.0)
    second = await _insert_run(postgresql, scheduled_at=start + timedelta(seconds=30))
    for _ in range(3):
        await _insert_result(postgresql, second, metric_value=9.0)
    await _fill_buckets(postgresql)

    response = await client.get(
        "/v1/results/timeline",
        params={
            "benchmark": "STT",
            # A 7-day custom range selects hourly buckets, so the two source
            # buckets (which the writer aligns to its scheduler cadence) merge.
            "since": (start - timedelta(days=6)).isoformat(),
            "until": (start + timedelta(days=1)).isoformat(),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["window"] is None
    assert body["aggregation"] == "average"
    assert body["bucket_seconds"] == 3600
    assert [point["value"] for point in body["points"]] == [pytest.approx(7.0)]
    assert body["points"][0]["aggregation_method"] == "mean_fallback"
    assert body["points"][0]["sample_count"] == 4


@pytest.mark.parametrize(
    ("coverage", "first_refs", "second_refs", "expected", "pooled"),
    [
        ("complete", 10, 30, 2.5, True),
        ("partial", 10, 30, 4.75, False),
        ("mismatched", 10, 30, 4.75, False),
        ("complete", 0, 30, 100 / 30, True),
        ("complete", 0, 0, 4.75, False),
    ],
)
async def test_normalized_timeline_average_uses_complete_wer_pool_or_fallback(
    client: AsyncClient,
    postgresql: Any,
    coverage: str,
    first_refs: int,
    second_refs: int,
    expected: float,
    pooled: bool,
) -> None:
    """Every included source must cover the primary clips before pooling counts."""
    bucket = datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    for offset, total, count, errors, refs in [
        (0, 10, 1, 1, first_refs),
        (15, 9, 3, 0, second_refs),
    ]:
        for key, value in [
            ("primary", total),
            ("substitution_count", errors),
            ("deletion_count", 0),
            ("insertion_count", 0),
            ("reference_words", refs),
        ]:
            if offset and key == "reference_words" and coverage == "partial":
                continue
            source_count = (
                2 if offset and key == "reference_words" and coverage == "mismatched" else count
            )
            await _insert_normalized_bucket(
                postgresql,
                dataset_id="stt-v2",
                value_key=key,
                value_sum=value,
                sample_count=source_count,
                bucket_at=bucket + timedelta(minutes=offset),
            )
    response = await client.get(
        "/v1/results/timeline",
        params={
            "benchmark": "STT",
            "dataset": "stt-v2",
            "since": (bucket - timedelta(days=6)).isoformat(),
            "until": (bucket + timedelta(days=1)).isoformat(),
        },
    )
    assert response.status_code == 200
    [point] = response.json()["points"]
    assert point["value"] == pytest.approx(expected)
    assert point["pooled_value"] == (pytest.approx(expected) if pooled else None)
    assert point["aggregation_method"] == ("ratio" if pooled else "mean_fallback")
    assert point["sample_count"] == 4


async def test_normalized_timeline_uses_registered_phonetic_ratio(
    client: AsyncClient, postgresql: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Timeline ratio rules use pooled operands and preserve incomplete sources as unavailable."""
    import psycopg

    from tests.api.conftest import _make_db_url

    rule = MetricValueContract(
        metric=Metric.RTF,
        version="v1",
        values=(
            MetricValueDefinition(
                key="primary", unit="percent", value_role=MetricValueRole.PRIMARY
            ),
            MetricValueDefinition(key="correct", unit="count", required=False),
            MetricValueDefinition(key="reference", unit="count", required=False),
        ),
        aggregation_method="ratio",
        numerator_keys=("correct",),
        denominator_key="reference",
        ratio_scale=100.0,
    )
    monkeypatch.setitem(TIMELINE_AGGREGATION_RULES, "PhoneticAccuracy", rule)
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = True
    start = datetime(2026, 8, 1, tzinfo=dt.UTC)
    rows = [
        (start, [("primary", 90, 1), ("correct", 9, 1), ("reference", 10, 1)]),
        (
            start + timedelta(minutes=15),
            [("primary", 50, 1), ("correct", 50, 1), ("reference", 100, 1)],
        ),
        (start + timedelta(hours=1), [("primary", 80, 1)]),
        (start + timedelta(hours=2), [("primary", 0, 1), ("correct", 0, 1), ("reference", 0, 1)]),
        (
            start + timedelta(hours=2, minutes=15),
            [("primary", 50, 1), ("correct", 5, 1), ("reference", 10, 1)],
        ),
        (start + timedelta(hours=3), [("primary", 0, 1), ("correct", 0, 1), ("reference", 0, 1)]),
        (start + timedelta(hours=4), [("primary", 80, 1), ("correct", 8, 1), ("reference", 10, 1)]),
        (start + timedelta(hours=5), [("primary", 80, 1), ("correct", 8, 2), ("reference", 10, 1)]),
        (
            start + timedelta(hours=5, minutes=15),
            [("primary", 160, 2), ("correct", 16, 1), ("reference", 20, 2)],
        ),
    ]
    async with await psycopg.AsyncConnection.connect(
        _make_db_url(postgresql), autocommit=True
    ) as conn:
        for bucket, values in rows:
            for key, value_sum, sample_count in values:
                await conn.execute(
                    """INSERT INTO benchmarks_v2.metric_values_by_bucket
                       (provider, model, benchmark, dataset_id, metric_type, metric_version,
                        evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50,
                        p75, max_value, value_sum, sample_count)
                       VALUES ('phonetic', 'accuracy-v1', 'STT', 'phonetic-v1',
                               'PhoneticAccuracy', 'v1', 'default', %s, %s, %s,
                               %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        key,
                        "invalid-unit"
                        if bucket == start + timedelta(hours=4) and key == "reference"
                        else ("count" if key != "primary" else "percent"),
                        bucket,
                        value_sum,
                        value_sum,
                        value_sum,
                        value_sum,
                        value_sum,
                        value_sum,
                        sample_count,
                    ),
                )
    response = await client.get(
        "/v1/results/timeline",
        params={
            "benchmark": "STT",
            "since": start.isoformat(),
            "until": (start + timedelta(hours=6)).isoformat(),
            "dataset": "phonetic-v1",
        },
    )
    assert response.status_code == 200
    points = [
        point for point in response.json()["points"] if point["metric_type"] == "PhoneticAccuracy"
    ]
    assert [point["value"] for point in points] == [
        pytest.approx(5900 / 110),
        None,
        pytest.approx(50),
        None,
        None,
        None,
    ]
    assert [point["aggregation_method"] for point in points] == [
        "ratio",
        "unavailable",
        "ratio",
        "unavailable",
        "unavailable",
        "unavailable",
    ]
    assert [point["sample_count"] for point in points] == [2, 1, 2, 1, 1, 3]


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


async def test_timeline_30d_averages_each_group_and_preserves_legacy_series(
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
        assert len(group) <= 62
        assert all(
            datetime.fromisoformat(p["scheduled_at"]).timestamp() % 14400 == 0 for p in group
        )

    assert response.json()["aggregation"] == "average"
    assert response.json()["bucket_seconds"] == 14400

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
    for group in legacy_groups.values():
        timestamps = {datetime.fromisoformat(p["scheduled_at"]) for p in group}
        assert now in timestamps and now - timedelta(hours=240) in timestamps
    assert min(p["p50"] for p in legacy_groups[("deepgram", "nova-3")]) == -100
    assert max(p["p50"] for p in legacy_groups[("deepgram", "nova-3")]) == 1000
    assert {"min_value", "p25", "p50", "p75", "max_value", "value_sum", "sample_count"} <= set(
        legacy.json()["series"][0]
    )


@pytest.mark.parametrize("normalized", [False, True])
async def test_timeline_ratio_without_fallback_is_unavailable_on_either_storage_path(
    client: AsyncClient, postgresql: Any, monkeypatch: pytest.MonkeyPatch, normalized: bool
) -> None:
    rule = MetricValueContract.model_validate(
        {
            **TIMELINE_AGGREGATION_RULES["WER"].model_dump(),
            "ratio_fallback": None,
        }
    )
    monkeypatch.setitem(TIMELINE_AGGREGATION_RULES, "WER", rule)
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.settings.normalized_dashboard_reads_enabled = normalized
    if normalized:
        await _insert_normalized_bucket(postgresql, dataset_id="__all__")
    else:
        run_id = await _insert_run(postgresql, scheduled_at=datetime.now(dt.UTC))
        await _insert_result(postgresql, run_id, metric_value=3)
        await _fill_buckets(postgresql)
    response = await client.get("/v1/results/timeline", params={"benchmark": "STT", "window": "7d"})
    assert response.status_code == 200
    [point] = response.json()["points"]
    assert point["value"] is None
    assert point["aggregation_method"] == "unavailable"
    assert point["sample_count"] > 0
