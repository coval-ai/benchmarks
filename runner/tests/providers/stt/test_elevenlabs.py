# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.elevenlabs (ElevenLabsSTTProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.stt.elevenlabs import ElevenLabsSTTProvider
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("elevenlabs", "events-success")
    provider = ElevenLabsSTTProvider(stt_settings)

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


@pytest.mark.asyncio
async def test_elevenlabs_joins_multiple_committed(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """Two distinct committed_transcript segments are joined in order."""
    events = load_fixture_events("elevenlabs", "events-multi-committed")
    provider = ElevenLabsSTTProvider(stt_settings)

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
async def test_elevenlabs_falls_back_to_last_partial(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """No committed transcript: the last partial is scored, not the longest."""
    events = [
        {"message_type": "session_started", "session_id": "x", "config": {}},
        {"message_type": "partial_transcript", "text": "hello there world"},
        {"message_type": "partial_transcript", "text": "hello world"},
    ]
    provider = ElevenLabsSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
    ):
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.complete_transcript == "hello world"
    assert result.audio_to_final_seconds is None


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


def test_build_websocket_url(stt_settings: Settings) -> None:
    url = ElevenLabsSTTProvider(stt_settings)._ws_url(16000)
    assert "elevenlabs.io" in url
    assert "scribe_v2_realtime" in url
    assert "audio_format=pcm_16000" in url


# ---------------------------------------------------------------------------
# Provider name + validation
# ---------------------------------------------------------------------------


def test_provider_name(stt_settings: Settings) -> None:
    assert ElevenLabsSTTProvider(stt_settings).name == "elevenlabs-scribe_v2_realtime"


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="elevenlabs_api_key is required"):
        ElevenLabsSTTProvider(make_settings(elevenlabs_api_key=None))


# ---------------------------------------------------------------------------
# Invalid sample rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_wrong_sample_rate(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = ElevenLabsSTTProvider(stt_settings)
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
@pytest.mark.parametrize("message_type", ["auth_error", "scribe_auth_error"])
async def test_elevenlabs_api_error(
    message_type: str, stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    error_events = [
        {"message_type": "session_started", "session_id": "x", "config": {}},
        {"message_type": message_type, "message": "Invalid API key"},
    ]
    provider = ElevenLabsSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(error_events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert message_type in result.error
    assert result.complete_transcript is None


# ---------------------------------------------------------------------------
# Failure path — empty stream (after session_started)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_empty_stream(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events = [
        {"message_type": "session_started", "session_id": "x", "config": {}},
    ]
    provider = ElevenLabsSTTProvider(stt_settings)

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

    assert result.complete_transcript is None
    assert result.ttft_seconds is None
