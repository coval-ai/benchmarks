# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.gradium (GradiumSTTProvider).

All tests use FakeWebSocket — no live network calls are made. The event
fixtures mirror the real Gradium wire protocol captured on a live socket:
each ``text`` message carries an additive word group for the current segment
(not a cumulative snapshot), so the full transcript is the in-order join.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.gradium import GradiumSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def make_provider() -> GradiumSTTProvider:
    return GradiumSTTProvider(api_key=SecretStr("test-key-gradium"))


def _fake_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.fixture(autouse=True)
def _fast_flush_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("coval_bench.providers.stt.gradium._FLUSH_WAIT_S", 0.05)


# ---------------------------------------------------------------------------
# Happy path — additive word groups joined in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("gradium", "events-success")
    provider = GradiumSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.gradium.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    assert result.audio_to_final_seconds is not None
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_gradium_commits_text_without_end_text(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """Each `text` word group is committed on arrival; `end_text` is not required.

    Pins the finalization contract: dropping every `end_text` from the stream
    leaves the assembled transcript unchanged, documenting that the provider
    commits on `text` rather than waiting for `end_text`.
    """
    events: list[Any] = [
        {"type": "text", "text": "hello world", "start_s": 0.5, "stream_id": 0},
        {"type": "text", "text": "how are you", "start_s": 1.2, "stream_id": 0},
        {"type": "flushed", "flush_id": 1},
        {"type": "end_of_stream"},
    ]
    provider = GradiumSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.gradium.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5


@pytest.mark.asyncio
async def test_gradium_closes_on_flushed_ack_not_blind_sleep(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `flushed` ack releases the close wait before the timeout cap.

    The cap is set high, so a prompt return proves the ack unblocked the wait.
    """
    monkeypatch.setattr("coval_bench.providers.stt.gradium._FLUSH_WAIT_S", 30.0)
    events: list[Any] = [
        {"type": "text", "text": "hello world", "start_s": 0.5, "stream_id": 0},
        {"type": "flushed", "flush_id": 1},
        {"type": "end_of_stream"},
    ]
    provider = GradiumSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.gradium.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        start = time.monotonic()
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )
        elapsed = time.monotonic() - start

    assert result.error is None
    assert result.complete_transcript == "hello world"
    # Ack releases the wait well before the 30s cap.
    assert elapsed < 5.0


# ---------------------------------------------------------------------------
# Provider name and model
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    assert make_provider().name == "gradium"


def test_provider_model() -> None:
    assert make_provider().model == "default"


# ---------------------------------------------------------------------------
# Invalid model
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Gradium STT model"):
        GradiumSTTProvider(api_key=SecretStr("k"), model="bad-model")


# ---------------------------------------------------------------------------
# Invalid sample rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_wrong_sample_rate(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = GradiumSTTProvider(api_key=fake_api_key)
    result = await provider.measure_ttft(
        audio_data=audio_pcm_bytes,
        channels=1,
        sample_width=2,
        sample_rate=8000,
    )

    assert result.error is not None
    assert "16 kHz" in result.error
    assert result.ttft_seconds is None


# ---------------------------------------------------------------------------
# Failure path — error event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("gradium", "events-error")
    provider = GradiumSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.gradium.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "Invalid or expired API key" in result.error
    assert result.complete_transcript is None


# ---------------------------------------------------------------------------
# Failure path — empty stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = GradiumSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.gradium.ws_client.connect",
        return_value=_fake_connect([]),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.complete_transcript is None
    assert result.ttft_seconds is None
