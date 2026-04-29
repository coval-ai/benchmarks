# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.elevenlabs (ElevenLabsSTTProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.elevenlabs import ElevenLabsSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def make_provider() -> ElevenLabsSTTProvider:
    return ElevenLabsSTTProvider(api_key=SecretStr("test-key-elevenlabs"))


def _fake_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("elevenlabs", "events-success")
    provider = ElevenLabsSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.elevenlabs.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
            audio_duration=3.0,
        )

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert result.complete_transcript is not None
    assert "hello world how are you" in result.complete_transcript


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


def test_build_websocket_url() -> None:
    p = make_provider()
    url = p._build_websocket_url()
    assert "elevenlabs.io" in url
    assert "scribe_v2_realtime" in url


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    p = make_provider()
    assert p.name == "elevenlabs-scribe_v2_realtime"


# ---------------------------------------------------------------------------
# Invalid model
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid ElevenLabs model"):
        ElevenLabsSTTProvider(api_key=SecretStr("k"), model="bad-model")


# ---------------------------------------------------------------------------
# Failure path — error event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_api_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    error_events = [
        {"message_type": "session_started", "session_id": "x", "config": {}},
        {"message_type": "scribe_auth_error", "message": "Invalid API key"},
    ]
    provider = ElevenLabsSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.elevenlabs.ws_client.connect",
        return_value=_fake_connect(error_events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "scribe_auth_error" in result.error
    assert result.complete_transcript is None


# ---------------------------------------------------------------------------
# Failure path — empty stream (after session_started)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = [
        {"message_type": "session_started", "session_id": "x", "config": {}},
    ]
    provider = ElevenLabsSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.elevenlabs.ws_client.connect",
        return_value=_fake_connect(events),
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
