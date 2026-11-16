# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/latency over HTTP."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from httpx import AsyncClient

from coval_bench.registries import RegisteredModel
from coval_bench.registries.benchmarks import Benchmark
from tests.api.conftest import EA_MODEL, EA_ORG, EA_PROVIDER, _make_db_url, add_models, bearer

NOW = datetime.now(UTC)


def _model_id(conn: Any, provider: str, model: str) -> int:
    row = conn.execute(
        "SELECT id FROM benchmarks_v2.models WHERE provider = %s AND model = %s", (provider, model)
    ).fetchone()
    return int(row[0])


def _probe(conn: Any, model_id: int, hours_ago: float, **fields: Any) -> None:
    at = NOW - timedelta(hours=hours_ago)
    error = fields.get("error")
    conn.execute(
        "INSERT INTO benchmarks_v2.probe_results"
        " (model_id, scheduled_at, sent_at, status, ttft_ms, ttfs_ms, ttfa_ms,"
        "  error, failure_origin, status_code)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            model_id,
            at,
            at,
            "failed" if error else "succeeded",
            fields.get("ttft"),
            fields.get("ttfs"),
            fields.get("ttfa"),
            error,
            fields.get("origin", "provider") if error else None,
            fields.get("status_code"),
        ),
    )


def _seed(postgresql: Any) -> tuple[int, int]:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        stt = _model_id(conn, "seed", "stt")
        tts = _model_id(conn, "seed", "tts-a")
        for hours_ago, ttft in enumerate((100.0, 200.0, 300.0), start=1):
            _probe(conn, stt, hours_ago, ttft=ttft, ttfs=ttft / 2)
        _probe(conn, stt, 4, error="timed out after 30s")
        _probe(conn, stt, 5, error="request failed", origin="runner", status_code=401)
        _probe(conn, stt, 6, error="clip did not load", origin="runner")
        _probe(conn, stt, 30, ttft=9000.0, ttfs=9000.0)
        _probe(conn, tts, 1, error="connection reset")
    return stt, tts


async def test_latency_is_percentiles_per_model_and_metric(
    client: AsyncClient, postgresql: Any
) -> None:
    stt, tts = _seed(postgresql)

    response = await client.get("/v3/latency")

    assert response.status_code == 200
    rows = {(row["model_id"], row["metric"]): row for row in response.json()["latency"]}
    assert set(rows) == {(stt, "TTFT"), (stt, "TTFS"), (tts, "TTFA")}
    assert rows[(stt, "TTFT")] == {
        "model_id": stt,
        "provider": "seed",
        "model": "stt",
        "metric": "TTFT",
        "unit": "milliseconds",
        "p50": 200.0,
        "p90": 280.0,
        "p95": 290.0,
        "mean": 200.0,
        "min": 100.0,
        "max": 300.0,
        "sample_count": 3,
        "failed_count": 1,
    }
    assert rows[(stt, "TTFS")]["p50"] == 100.0
    down = rows[(tts, "TTFA")]
    assert (down["p50"], down["sample_count"], down["failed_count"]) == (None, 0, 1)


async def test_latency_filters_by_modality_metric_and_window(
    client: AsyncClient, postgresql: Any
) -> None:
    stt, tts = _seed(postgresql)

    async def rows(query: str) -> dict[tuple[int, str], Any]:
        body = (await client.get(f"/v3/latency?{query}")).json()
        return {(row["model_id"], row["metric"]): row for row in body["latency"]}

    assert set(await rows("modality=TTS")) == {(tts, "TTFA")}
    assert set(await rows("metric=TTFS")) == {(stt, "TTFS")}
    week = (await rows("window=7d"))[(stt, "TTFT")]
    assert (week["sample_count"], week["max"]) == (4, 9000.0)


async def test_an_unpublished_model_is_only_in_its_own_orgs_latency(
    client: AsyncClient, postgresql: Any
) -> None:
    add_models(
        postgresql,
        RegisteredModel(
            benchmark=Benchmark.STT,
            provider=EA_PROVIDER,
            model=EA_MODEL,
            collected=True,
            published=False,
        ),
    )
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        hidden = _model_id(conn, EA_PROVIDER, EA_MODEL)
        _probe(conn, hidden, 1, ttft=150.0, ttfs=80.0)

    public = await client.get("/v3/latency")
    partner = await client.get("/v3/latency", headers=bearer(sub="u", org_id=EA_ORG))

    assert hidden not in {row["model_id"] for row in public.json()["latency"]}
    assert hidden in {row["model_id"] for row in partner.json()["latency"]}
    assert partner.headers["Cache-Control"] == "private, no-store"


async def test_a_bad_parameter_is_named_in_the_error(client: AsyncClient) -> None:
    response = await client.get("/v3/latency?modality=S2S")

    assert response.status_code == 400
    error = response.json()["error"]
    assert (error["code"], error["param"]) == ("invalid_parameter", "modality")
    assert "STT" in error["message"]


async def test_an_unknown_v3_path_is_a_not_found_error(client: AsyncClient) -> None:
    response = await client.get("/v3/nope")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
