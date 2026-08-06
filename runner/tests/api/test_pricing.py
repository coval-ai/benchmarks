# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GET /v1/pricing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from httpx import AsyncClient

from tests.api.conftest import _insert_result, _insert_run, _make_db_url


async def _insert_price(
    postgresql: Any,
    *,
    provider: str,
    model: str,
    benchmark: str,
    billing_unit: str,
    rate_usd: str,
    effective_at: datetime,
    superseded_at: datetime | None = None,
    plan_assumption: str | None = None,
) -> None:
    aconn = await psycopg.AsyncConnection.connect(_make_db_url(postgresql), autocommit=True)
    try:
        await aconn.execute(
            "INSERT INTO benchmarks_v2.model_pricing"
            " (provider, model, benchmark, billing_unit, rate_usd, plan_assumption,"
            "  effective_at, superseded_at, source_url, as_of, updated_by)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s,"
            "  'https://example.com/pricing', '2026-08-06', 'human')",
            (
                provider,
                model,
                benchmark,
                billing_unit,
                rate_usd,
                plan_assumption,
                effective_at,
                superseded_at,
            ),
        )
    finally:
        await aconn.close()


async def test_benchmark_required(client: AsyncClient) -> None:
    assert (await client.get("/v1/pricing")).status_code == 422


async def test_empty_db_returns_no_entries(client: AsyncClient) -> None:
    response = await client.get("/v1/pricing", params={"benchmark": "TTS"})
    assert response.status_code == 200
    body = response.json()
    assert body["unit_label"] == "USD per 1M characters"
    assert body["entries"] == []


async def test_duration_billed_model_normalizes_from_list_price(
    client: AsyncClient, postgresql: Any
) -> None:
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    t1 = datetime(2026, 8, 1, tzinfo=UTC)
    # Superseded history row (was $0.004/min) then the current $0.006/min.
    await _insert_price(
        postgresql,
        provider="mistral",
        model="voxtral-mini-transcribe-realtime-2602",
        benchmark="STT",
        billing_unit="per_minute",
        rate_usd="0.004",
        effective_at=t0,
        superseded_at=t1,
    )
    await _insert_price(
        postgresql,
        provider="mistral",
        model="voxtral-mini-transcribe-realtime-2602",
        benchmark="STT",
        billing_unit="per_minute",
        rate_usd="0.006",
        effective_at=t1,
    )

    response = await client.get("/v1/pricing", params={"benchmark": "STT"})
    assert response.status_code == 200
    body = response.json()
    assert body["unit_label"] == "USD per 1,000 minutes"
    assert len(body["entries"]) == 1
    entry = body["entries"][0]
    assert entry["provider"] == "mistral"
    assert entry["normalized_usd"] == pytest.approx(6.0)
    assert entry["basis"] == "list_price"
    assert entry["as_of"] == "2026-08-06"
    assert entry["source_url"] == "https://example.com/pricing"
    assert entry["native_rates"] == [
        {"billing_unit": "per_minute", "rate_usd": 0.006, "plan_assumption": None}
    ]
    # History: the old rate normalized too, oldest first.
    history = entry["history"]
    assert len(history) == 2
    assert history[0]["normalized_usd"] == pytest.approx(4.0)
    assert history[0]["superseded_at"] is not None
    assert history[1]["normalized_usd"] == pytest.approx(6.0)
    assert history[1]["superseded_at"] is None


async def test_token_billed_model_uses_measured_conversion(
    client: AsyncClient, postgresql: Any
) -> None:
    now = datetime.now(tz=UTC)
    for unit, rate in (("per_1m_tokens_input", "2.50"), ("per_1m_tokens_output", "10.00")):
        await _insert_price(
            postgresql,
            provider="openai",
            model="gpt-4o-transcribe",
            benchmark="STT",
            billing_unit=unit,
            rate_usd=rate,
            effective_at=now - timedelta(days=30),
        )
    # 50 anchor rows in the 7d window: 1000 in-tokens + 100 out-tokens per
    # 60s item → 1000 in-tokens/min, 100 out-tokens/min measured.
    run_id = await _insert_run(postgresql)
    for _ in range(50):
        await _insert_result(
            postgresql,
            run_id,
            provider="openai",
            model="gpt-4o-transcribe",
            metric_type="AudioToFinal",
            metric_value=1.0,
            input_tokens=1000,
            output_tokens=100,
            audio_seconds_in=60.0,
        )

    response = await client.get("/v1/pricing", params={"benchmark": "STT"})
    assert response.status_code == 200
    entries = response.json()["entries"]
    assert len(entries) == 1
    entry = entries[0]
    # (2.5/1e6 * 1000 + 10/1e6 * 100) * 1000 = $3.50 per 1k min
    assert entry["normalized_usd"] == pytest.approx(3.5)
    assert entry["basis"] == "list_price_measured_conversion"
    assert entry["conversion"]["in_tokens_per_min"] == pytest.approx(1000.0)
    assert entry["conversion"]["out_tokens_per_min"] == pytest.approx(100.0)
    assert entry["conversion"]["sample_count"] == 50
    assert entry["conversion"]["window"] == "7d"
    assert {r["billing_unit"] for r in entry["native_rates"]} == {
        "per_1m_tokens_input",
        "per_1m_tokens_output",
    }


async def test_token_billed_model_without_conversion_serves_native_only(
    client: AsyncClient, postgresql: Any
) -> None:
    await _insert_price(
        postgresql,
        provider="openai",
        model="gpt-4o-transcribe",
        benchmark="STT",
        billing_unit="per_1m_tokens_input",
        rate_usd="2.50",
        effective_at=datetime.now(tz=UTC) - timedelta(days=1),
    )
    response = await client.get("/v1/pricing", params={"benchmark": "STT"})
    entries = response.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["normalized_usd"] is None
    assert entries[0]["basis"] is None
    assert entries[0]["conversion"] is None
    assert entries[0]["native_rates"][0]["rate_usd"] == pytest.approx(2.5)


async def test_history_periods_chain_when_token_pair_staggers(
    client: AsyncClient, postgresql: Any
) -> None:
    """A pair whose units changed at different times still yields contiguous periods."""
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    t1 = datetime(2026, 8, 1, tzinfo=UTC)
    await _insert_price(
        postgresql,
        provider="openai",
        model="gpt-4o-transcribe",
        benchmark="STT",
        billing_unit="per_1m_tokens_input",
        rate_usd="2.50",
        effective_at=t0,
    )
    await _insert_price(
        postgresql,
        provider="openai",
        model="gpt-4o-transcribe",
        benchmark="STT",
        billing_unit="per_1m_tokens_output",
        rate_usd="10.00",
        effective_at=t1,
    )
    response = await client.get("/v1/pricing", params={"benchmark": "STT"})
    history = response.json()["entries"][0]["history"]
    assert len(history) == 2
    assert history[0]["superseded_at"] == history[1]["effective_at"]
    assert history[1]["superseded_at"] is None


async def test_unregistered_model_absent(client: AsyncClient, postgresql: Any) -> None:
    """Rows for models not in the active registry never surface."""
    await _insert_price(
        postgresql,
        provider="openai",
        model="whisper-1-judge",
        benchmark="TTS",
        billing_unit="per_minute",
        rate_usd="0.006",
        effective_at=datetime.now(tz=UTC) - timedelta(days=1),
    )
    response = await client.get("/v1/pricing", params={"benchmark": "TTS"})
    assert response.json()["entries"] == []
