# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.deepgram (DeepgramProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.deepgram import DeepgramProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_provider(model: str = "nova-3") -> DeepgramProvider:
    return DeepgramProvider(api_key=SecretStr("test-key-deepgram"), model=model)


def _fake_connect(events: list[Any]) -> Any:
    """Return an async context manager that yields a FakeWebSocket."""
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Happy path — nova-3
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_nova3_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("deepgram", "events-success")
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
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
    assert "hello" in result.first_token_content.lower()
    assert result.complete_transcript is not None
    assert "hello" in result.complete_transcript.lower()
    # VAD: 1 SpeechStarted in fixture
    assert result.vad_events_count == 1
    assert result.vad_first_detected is not None


# ---------------------------------------------------------------------------
# Happy path — default model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_default_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("deepgram", "events-success")
    provider = DeepgramProvider(api_key=fake_api_key, model="default")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
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
    assert result.complete_transcript is not None


# ---------------------------------------------------------------------------
# URL builder sanity
# ---------------------------------------------------------------------------


def test_build_websocket_url_nova3() -> None:
    p = make_provider("nova-3")
    url = p._build_websocket_url(16000, 1)
    assert "nova-3" in url
    assert "api.deepgram.com" in url


def test_build_websocket_url_flux() -> None:
    p = make_provider("flux-general-en")
    url = p._build_websocket_url(16000, 1)
    assert "preview.deepgram.com" in url
    assert "flux-general-en" in url


# ---------------------------------------------------------------------------
# Invalid model
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Deepgram model"):
        DeepgramProvider(api_key=SecretStr("k"), model="nova-99")


# ---------------------------------------------------------------------------
# Failure path — empty event stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
        return_value=_fake_connect([]),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    # No transcript events → no complete transcript
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


# ---------------------------------------------------------------------------
# Failure path — error event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("deepgram", "events-error")
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    # Error events don't produce a transcript
    assert result.complete_transcript is None


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


def test_provider_name_with_model() -> None:
    p = make_provider("nova-2")
    assert p.name == "deepgram-nova-2"


def test_provider_name_default() -> None:
    p = make_provider("default")
    assert p.name == "deepgram"
