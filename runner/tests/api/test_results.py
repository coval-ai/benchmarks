# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GET /v1/results."""

from __future__ import annotations

import datetime as dt
from datetime import datetime, timedelta
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


# ---------------------------------------------------------------------------
# D2: status field on ResultOut
# ---------------------------------------------------------------------------


async def test_status_field_present(client: AsyncClient, postgresql: Any) -> None:
    """Response rows include 'status' key matching parent run status (uppercase)."""
    run_id = await _insert_run(postgresql, status="succeeded")
    await _insert_result(postgresql, run_id)

    response = await client.get("/v1/results")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert "status" in results[0]
    assert results[0]["status"] == "SUCCEEDED"


# ---------------------------------------------------------------------------
# D1: window parameter tests
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=dt.UTC)


async def test_window_default_7d(client: AsyncClient, postgresql: Any) -> None:
    """Default window (7d): rows at -1d and -10d; only -1d is within window."""
    run_id = await _insert_run(postgresql)
    now = _now()
    # Row at -1 day (within 7d window)
    await _insert_result(postgresql, run_id, created_at=now - timedelta(days=1), model="nova-3")
    # Row at -10 days (outside 7d window)
    await _insert_result(postgresql, run_id, created_at=now - timedelta(days=10), model="nova-2")
    # Row at -40 days (outside 7d window)
    await _insert_result(postgresql, run_id, created_at=now - timedelta(days=40), model="nova-2")

    response = await client.get("/v1/results")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["model"] == "nova-3"


async def test_window_24h(client: AsyncClient, postgresql: Any) -> None:
    """window=24h: row at -1h returned; row at -2d excluded."""
    run_id = await _insert_run(postgresql)
    now = _now()
    await _insert_result(postgresql, run_id, created_at=now - timedelta(hours=1), model="nova-3")
    await _insert_result(postgresql, run_id, created_at=now - timedelta(days=2), model="nova-2")

    response = await client.get("/v1/results", params={"window": "24h"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["model"] == "nova-3"


async def test_window_30d(client: AsyncClient, postgresql: Any) -> None:
    """window=30d: rows at -1d and -10d returned; -40d excluded."""
    run_id = await _insert_run(postgresql)
    now = _now()
    await _insert_result(postgresql, run_id, created_at=now - timedelta(days=1), model="nova-3")
    await _insert_result(postgresql, run_id, created_at=now - timedelta(days=10), model="nova-2")
    await _insert_result(
        postgresql, run_id, created_at=now - timedelta(days=40), model="flux-general-en"
    )

    response = await client.get("/v1/results", params={"window": "30d"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    models = {r["model"] for r in results}
    assert "nova-3" in models
    assert "nova-2" in models
    assert "flux-general-en" not in models


async def test_window_with_since_returns_400(client: AsyncClient) -> None:
    """window + since returns 400."""
    response = await client.get(
        "/v1/results",
        params={"window": "7d", "since": "2024-01-01T00:00:00Z"},
    )
    assert response.status_code == 400
    assert "window" in response.json()["detail"].lower()


async def test_window_with_until_returns_400(client: AsyncClient) -> None:
    """window + until returns 400."""
    response = await client.get(
        "/v1/results",
        params={"window": "7d", "until": "2099-01-01T00:00:00Z"},
    )
    assert response.status_code == 400
    assert "window" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# D1: include_failed parameter tests
# ---------------------------------------------------------------------------


async def test_include_failed_default_excludes_failed_runs(
    client: AsyncClient, postgresql: Any
) -> None:
    """Default include_failed=false: result on a FAILED run is excluded."""
    run_failed = await _insert_run(postgresql, status="failed")
    run_ok = await _insert_run(postgresql, status="succeeded")
    await _insert_result(postgresql, run_failed, model="nova-2")
    await _insert_result(postgresql, run_ok, model="nova-3")

    response = await client.get("/v1/results")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["model"] == "nova-3"
    assert results[0]["status"] == "SUCCEEDED"


async def test_include_failed_true_includes_failed_runs(
    client: AsyncClient, postgresql: Any
) -> None:
    """include_failed=true: results from FAILED runs are included."""
    run_failed = await _insert_run(postgresql, status="failed")
    run_ok = await _insert_run(postgresql, status="succeeded")
    await _insert_result(postgresql, run_failed, model="nova-2")
    await _insert_result(postgresql, run_ok, model="nova-3")

    response = await client.get("/v1/results", params={"include_failed": "true"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    statuses = {r["status"] for r in results}
    assert "SUCCEEDED" in statuses
    assert "FAILED" in statuses


async def test_partial_run_results_included_by_default(
    client: AsyncClient, postgresql: Any
) -> None:
    """PARTIAL runs are included by default (include_failed=false)."""
    run_partial = await _insert_run(postgresql, status="partial")
    await _insert_result(postgresql, run_partial, model="nova-3")

    response = await client.get("/v1/results")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "PARTIAL"


# ---------------------------------------------------------------------------
# D1: metric_type alias tests
# ---------------------------------------------------------------------------


async def test_metric_and_metric_type_alias_match(client: AsyncClient, postgresql: Any) -> None:
    """metric=WER&metric_type=WER (equal) → 200."""
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, metric_type="WER")

    response = await client.get(
        "/v1/results",
        params={"metric": "WER", "metric_type": "WER"},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1


async def test_metric_and_metric_type_alias_mismatch(client: AsyncClient) -> None:
    """metric=WER&metric_type=TTFA (different) → 400."""
    response = await client.get(
        "/v1/results",
        params={"metric": "WER", "metric_type": "TTFA"},
    )
    assert response.status_code == 400
    assert "alias" in response.json()["detail"].lower()


async def test_metric_type_only(client: AsyncClient, postgresql: Any) -> None:
    """metric_type=WER alone works the same as metric=WER."""
    run_id = await _insert_run(postgresql)
    await _insert_result(postgresql, run_id, metric_type="WER")
    await _insert_result(postgresql, run_id, metric_type="TTFA")

    response = await client.get("/v1/results", params={"metric_type": "WER"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["metric_type"] == "WER"


# ---------------------------------------------------------------------------
# D1: limit parameter tests
# ---------------------------------------------------------------------------


async def test_limit_default_100000(client: AsyncClient, postgresql: Any) -> None:
    """Default limit is 100000 — seeding 150 rows returns all 150."""
    run_id = await _insert_run(postgresql)
    for i in range(150):
        await _insert_result(postgresql, run_id, model=f"model-{i:03d}")

    response = await client.get("/v1/results")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 150


async def test_limit_above_cap_rejected(client: AsyncClient) -> None:
    """limit=100001 → 422 (above cap)."""
    response = await client.get("/v1/results", params={"limit": 100001})
    assert response.status_code == 422


async def test_limit_zero_rejected(client: AsyncClient) -> None:
    """limit=0 → 422 (below minimum)."""
    response = await client.get("/v1/results", params={"limit": 0})
    assert response.status_code == 422
