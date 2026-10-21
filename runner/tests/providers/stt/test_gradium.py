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
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.gradium import GradiumSTTProvider
from tests.providers.stt.conftest import fake_connect, load_fixture_events, make_settings


def make_provider() -> GradiumSTTProvider:
    return GradiumSTTProvider(make_settings())


# ---------------------------------------------------------------------------
# Happy path — additive word groups joined in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("gradium", "events-success")
    provider = GradiumSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
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
    stt_settings: Settings, audio_pcm_bytes: bytes
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
    provider = GradiumSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
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
    stt_settings: Settings, audio_pcm_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `flushed` ack releases the close wait before the timeout cap.

    The cap is set high, so a prompt return proves the ack unblocked the wait.
    """
    monkeypatch.setattr(GradiumSTTProvider, "_FINAL_WAIT_S", 30.0)
    events: list[Any] = [
        {"type": "text", "text": "hello world", "start_s": 0.5, "stream_id": 0},
        {"type": "flushed", "flush_id": 1},
        {"type": "end_of_stream"},
    ]
    provider = GradiumSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
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


@pytest.mark.asyncio
async def test_gradium_wire_shape(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """Setup frame first, base64 audio frames, then flush and end_of_stream."""
    import base64
    import json

    sent: list[Any] = []
    events: list[Any] = [{"type": "flushed", "flush_id": 1}, {"type": "end_of_stream"}]
    provider = GradiumSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events, sent),
    ) as connect:
        await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert connect.call_args.kwargs["additional_headers"] == {
        "x-api-key": "test-api-key-00000000000000000000"
    }
    frames = [json.loads(m) for m in sent]
    assert frames[0]["type"] == "setup"
    assert frames[0]["input_format"] == "pcm_16000"
    audio = b"".join(base64.b64decode(f["audio"]) for f in frames if f["type"] == "audio")
    assert audio == audio_pcm_bytes
    assert [f["type"] for f in frames[-2:]] == ["flush", "end_of_stream"]


# ---------------------------------------------------------------------------
# Provider name and model
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    assert make_provider().name == "gradium"


def test_provider_model() -> None:
    assert make_provider().model == "default"


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="gradium_api_key is required"):
        GradiumSTTProvider(make_settings(gradium_api_key=None))


# ---------------------------------------------------------------------------
# Invalid sample rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_wrong_sample_rate(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = GradiumSTTProvider(stt_settings)
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
async def test_gradium_error_event(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("gradium", "events-error")
    provider = GradiumSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
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
async def test_gradium_empty_stream(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = GradiumSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect([]),
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
