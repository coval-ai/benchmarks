# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GET /v1/leaderboard."""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient

from coval_bench.api.common import MIN_SCORED_SAMPLES
from tests.api.conftest import _insert_run, _insert_value, _publish


async def test_24h_window_sorted_ascending(client: AsyncClient, postgresql: Any) -> None:
    """window=24h returns entries sorted ascending by avg."""
    run_id = await _insert_run(postgresql)
    # Two providers with different WER values
    await _insert_value(postgresql, run_id, 5.0)
    await _insert_value(postgresql, run_id, 8.0, model="nova-2")
    await _publish(postgresql)

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


async def test_ttft_stt_7d_window(client: AsyncClient, postgresql: Any) -> None:
    """window=7d serves TTFT+STT from the 7d snapshot."""
    run_id = await _insert_run(postgresql)
    await _insert_value(postgresql, run_id, 120.0, metric_type="TTFT")
    await _publish(postgresql)

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


async def test_30d_window(client: AsyncClient, postgresql: Any) -> None:
    """window=30d serves from the 30d snapshot."""
    run_id = await _insert_run(postgresql)
    await _insert_value(postgresql, run_id, 6.0, model="nova-2")
    await _publish(postgresql)
    response = await client.get(
        "/v1/leaderboard",
        params={"metric": "WER", "benchmark": "STT", "window": "30d"},
    )
    assert response.status_code == 200
    assert len(response.json()["entries"]) == 1


async def test_excluded_metric_rows_hidden(client: AsyncClient, postgresql: Any) -> None:
    """Historical rows for METRIC_EXCLUSIONS pairs are hidden from the leaderboard."""
    run_id = await _insert_run(postgresql)
    await _insert_value(
        postgresql,
        run_id,
        0.4,
        provider="assemblyai",
        model="universal-streaming",
        metric_type="TTFS",
    )
    await _insert_value(postgresql, run_id, 0.6, metric_type="TTFS")
    await _publish(postgresql)

    response = await client.get(
        "/v1/leaderboard", params={"metric": "TTFS", "benchmark": "STT", "window": "24h"}
    )
    assert response.status_code == 200
    entries = response.json()["entries"]
    assert [(e["provider"], e["model"]) for e in entries] == [("deepgram", "nova-3")]


async def test_s2s_leaderboard_uses_the_bank_primary_dataset(
    client: AsyncClient, postgresql: Any
) -> None:
    """The headline S2S board is the Ultra Bank set; the frozen dental rows stay out."""
    primary_run = await _insert_run(postgresql, dataset_id="s2s-bank-v1")
    dental_run = await _insert_run(postgresql, dataset_id="s2s-dental-v1")
    await _insert_value(
        postgresql,
        primary_run,
        1200.0,
        provider="openai",
        model="gpt-realtime",
        metric_type="V2V",
        benchmark="S2S",
        dataset_id="s2s-bank-v1",
    )
    await _insert_value(
        postgresql,
        dental_run,
        900.0,
        provider="google",
        model="gemini-live",
        metric_type="V2V",
        benchmark="S2S",
        dataset_id="s2s-dental-v1",
    )
    await _publish(postgresql)

    response = await client.get("/v1/leaderboard", params={"metric": "V2V", "benchmark": "S2S"})

    assert response.status_code == 200
    assert [(e["provider"], e["model"]) for e in response.json()["entries"]] == [
        ("openai", "gpt-realtime")
    ]


async def test_ttft_llm_uses_the_bank_primary_dataset(client: AsyncClient, postgresql: Any) -> None:
    """The LLM board reads the text bank dataset; rows under other datasets stay out."""
    bank_run = await _insert_run(postgresql, dataset_id="llm-bank-v1")
    other_run = await _insert_run(postgresql, dataset_id="llm-scratch-v1")
    await _insert_value(
        postgresql,
        bank_run,
        0.42,
        provider="phonely",
        model="phonely-agent",
        metric_type="TTFT",
        benchmark="LLM",
        dataset_id="llm-bank-v1",
    )
    await _insert_value(
        postgresql,
        other_run,
        0.10,
        provider="acme",
        model="chat-1",
        metric_type="TTFT",
        benchmark="LLM",
        dataset_id="llm-scratch-v1",
    )
    await _publish(postgresql)

    response = await client.get("/v1/leaderboard", params={"metric": "TTFT", "benchmark": "LLM"})
    assert response.status_code == 200
    assert [(e["provider"], e["model"]) for e in response.json()["entries"]] == [
        ("phonely", "phonely-agent")
    ]


async def test_v2v_llm_incompatible(client: AsyncClient) -> None:
    """A text agent has no voice-to-voice latency."""
    response = await client.get("/v1/leaderboard", params={"metric": "V2V", "benchmark": "LLM"})
    assert response.status_code == 400


async def test_thin_entry_flagged_and_sunk_below_ranked(
    client: AsyncClient, postgresql: Any
) -> None:
    """A model scored fewer times than the floor can't outrank a well-sampled one.

    The view orders by value alone, so a single fast measurement would otherwise
    lead the board.
    """
    run_id = await _insert_run(postgresql)
    # One lucky, very fast sample — would sort first on value.
    await _insert_value(postgresql, run_id, 0.0, provider="fishaudio", model="s1")
    for _ in range(MIN_SCORED_SAMPLES["STT"]):
        await _insert_value(postgresql, run_id, 9.0)
    await _publish(postgresql)

    response = await client.get(
        "/v1/leaderboard", params={"metric": "WER", "benchmark": "STT", "window": "24h"}
    )
    assert response.status_code == 200
    entries = response.json()["entries"]

    assert [(e["model"], e["insufficient_samples"]) for e in entries] == [
        ("nova-3", False),
        ("s1", True),
    ]
    # The measurement is still reported, just not presented as rankable.
    thin = entries[1]
    assert thin["avg"] == 0.0
    assert thin["n"] == 1


async def test_entry_at_the_floor_is_not_flagged(client: AsyncClient, postgresql: Any) -> None:
    """Exactly the floor's worth of samples counts as enough."""
    run_id = await _insert_run(postgresql)
    for _ in range(MIN_SCORED_SAMPLES["STT"]):
        await _insert_value(postgresql, run_id, 4.0)
    await _publish(postgresql)

    response = await client.get(
        "/v1/leaderboard", params={"metric": "WER", "benchmark": "STT", "window": "24h"}
    )
    entries = response.json()["entries"]
    assert [(e["n"], e["insufficient_samples"]) for e in entries] == [
        (MIN_SCORED_SAMPLES["STT"], False)
    ]
