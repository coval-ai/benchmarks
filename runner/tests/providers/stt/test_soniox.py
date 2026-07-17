# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.soniox (SonioxSTTProvider).

All tests use FakeWebSocket — no live network calls are made. The event
fixtures mirror the Soniox real-time wire protocol: each message carries a
``tokens`` array where ``is_final`` marks committed tokens. Token ``text``
already carries its own spacing, so the transcript is the direct concatenation
of final-token text.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.soniox import SonioxSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def make_provider() -> SonioxSTTProvider:
    return SonioxSTTProvider(api_key=SecretStr("test-key-soniox"))


def _fake_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Happy path — final tokens concatenated in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soniox_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("soniox", "events-success")
    provider = SonioxSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.soniox.ws_client.connect",
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
async def test_soniox_excludes_interim_tokens(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """Only ``is_final`` tokens form the transcript; interim tokens are dropped.

    The interim ("wrong") tokens differ from the committed text, so a leak into
    the final transcript would change the assertion.
    """
    events: list[Any] = [
        {"tokens": [{"text": "wrong guess", "is_final": False}]},
        {"tokens": [{"text": "hello", "is_final": True}, {"text": " worng", "is_final": False}]},
        {"tokens": [{"text": " world", "is_final": True}]},
        {"finished": True},
    ]
    provider = SonioxSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.soniox.ws_client.connect",
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
    assert make_provider().name == "soniox"


def test_provider_model() -> None:
    assert make_provider().model == "stt-rt-v5"


def test_stt_rt_v4_still_supported() -> None:
    assert SonioxSTTProvider(api_key=SecretStr("k"), model="stt-rt-v4").model == "stt-rt-v4"


# ---------------------------------------------------------------------------
# Invalid construction
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Soniox STT model"):
        SonioxSTTProvider(api_key=SecretStr("k"), model="stt-rt-preview")


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="soniox_api_key is required"):
        SonioxSTTProvider(api_key=None)


# ---------------------------------------------------------------------------
# Invalid sample rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soniox_wrong_sample_rate(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = SonioxSTTProvider(api_key=fake_api_key)
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
async def test_soniox_rejects_non_mono_or_non_16bit(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """The config hardcodes mono 16-bit, so non-matching input is rejected upfront."""
    provider = SonioxSTTProvider(api_key=fake_api_key)
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
async def test_soniox_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("soniox", "events-error")
    provider = SonioxSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.soniox.ws_client.connect",
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
async def test_soniox_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = SonioxSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.soniox.ws_client.connect",
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


# ---------------------------------------------------------------------------
# Failure path — websocket transport exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soniox_connection_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """A connect failure surfaces as result.error, not a silent empty success."""
    provider = SonioxSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.soniox.ws_client.connect",
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
async def test_soniox_surfaces_send_failure(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A send failure inside the streaming task is surfaced, not swallowed by gather."""

    def _raise_on_audio(msg: object) -> None:
        if isinstance(msg, (bytes, bytearray)):
            raise RuntimeError("send boom")

    ws = FakeWebSocket([], on_send=_raise_on_audio)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    provider = SonioxSTTProvider(api_key=fake_api_key)
    with patch("coval_bench.providers.stt.soniox.ws_client.connect", return_value=cm):
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
