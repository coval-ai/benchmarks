# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.inworld (InworldSTTProvider).

All tests use FakeWebSocket — no live network calls are made. The event
fixtures mirror the Inworld real-time wire protocol: each message nests a
``result.transcription`` object where ``isFinal`` marks committed segments.
Final segments are whole phrases, so the transcript joins them with spaces.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.inworld import InworldSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def make_provider() -> InworldSTTProvider:
    return InworldSTTProvider(api_key=SecretStr("test-key-inworld"))


def _fake_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Happy path — final segments joined in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inworld_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("inworld", "events-success")
    provider = InworldSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.inworld.ws_client.connect",
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
async def test_inworld_excludes_interim_segments(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """Only ``isFinal`` segments form the transcript; interim ones are dropped.

    The interim ("wrong") segments differ from the committed text, so a leak
    into the final transcript would change the assertion.
    """
    events: list[Any] = [
        {"result": {"transcription": {"transcript": "wrong guess", "isFinal": False}}},
        {"result": {"transcription": {"transcript": "hello", "isFinal": True}}},
        {"result": {"transcription": {"transcript": "worng", "isFinal": False}}},
        {"result": {"transcription": {"transcript": "world", "isFinal": True}}},
    ]
    provider = InworldSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.inworld.ws_client.connect",
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
    assert result.complete_transcript == "hello world"
    assert result.word_count == 2


# ---------------------------------------------------------------------------
# Provider name and model
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    assert make_provider().name == "inworld"


def test_provider_model() -> None:
    assert make_provider().model == "inworld-stt-1"


# ---------------------------------------------------------------------------
# Invalid construction
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Inworld STT model"):
        InworldSTTProvider(api_key=SecretStr("k"), model="inworld-stt-preview")


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="inworld_api_key is required"):
        InworldSTTProvider(api_key=None)


def test_blank_api_key_raises() -> None:
    with pytest.raises(ValueError, match="inworld_api_key is required"):
        InworldSTTProvider(api_key=SecretStr("   "))


# ---------------------------------------------------------------------------
# Invalid sample rate / format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inworld_wrong_sample_rate(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = InworldSTTProvider(api_key=fake_api_key)
    result = await provider.measure_ttft(
        audio_data=audio_pcm_bytes,
        channels=1,
        sample_width=2,
        sample_rate=8000,
    )

    assert result.error is not None
    assert "16 kHz" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_inworld_rejects_non_mono_or_non_16bit(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """The config hardcodes mono 16-bit, so non-matching input is rejected upfront."""
    provider = InworldSTTProvider(api_key=fake_api_key)
    result = await provider.measure_ttft(
        audio_data=audio_pcm_bytes,
        channels=2,
        sample_width=2,
        sample_rate=16000,
    )

    assert result.error is not None
    assert "mono 16-bit" in result.error
    assert result.ttft_seconds is None


# ---------------------------------------------------------------------------
# Failure path — error frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inworld_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("inworld", "events-error")
    provider = InworldSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.inworld.ws_client.connect",
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
async def test_inworld_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """A stream that never yields a final transcription is a failure, not an empty success."""
    provider = InworldSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.inworld.ws_client.connect",
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
    assert result.error is not None
    assert "before a final transcription" in result.error


# ---------------------------------------------------------------------------
# Failure path — websocket transport exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inworld_connection_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """A connect failure surfaces as result.error, not a silent empty success."""
    provider = InworldSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.inworld.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_inworld_surfaces_send_failure(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A send failure inside the streaming task is surfaced, not swallowed by gather."""

    def _raise_on_audio(msg: object) -> None:
        if isinstance(msg, str) and "audioChunk" in msg:
            raise RuntimeError("send boom")

    ws = FakeWebSocket([], on_send=_raise_on_audio)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    provider = InworldSTTProvider(api_key=fake_api_key)
    with patch("coval_bench.providers.stt.inworld.ws_client.connect", return_value=cm):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "send boom" in result.error
    assert result.complete_transcript is None
