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

import json
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.soniox import SonioxSTTProvider
from tests.providers.fakes import FakeWebSocket, fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings


def make_provider() -> SonioxSTTProvider:
    return SonioxSTTProvider(make_settings(soniox_api_key=SecretStr("test-key-soniox")))


# ---------------------------------------------------------------------------
# Happy path — final tokens concatenated in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soniox_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("soniox", "events-success")
    provider = SonioxSTTProvider(stt_settings)

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
async def test_soniox_authenticates_with_header(audio_pcm_bytes: bytes) -> None:
    sent: list[object] = []
    ws = FakeWebSocket([{"finished": True}], on_send=sent.append)
    cm = fake_connect(ws)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect", return_value=cm
    ) as connect:
        await make_provider().measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert connect.call_args.kwargs["additional_headers"] == {
        "Authorization": "Bearer test-key-soniox"
    }
    config = json.loads(next(m for m in sent if isinstance(m, str)))
    assert "api_key" not in config
    assert config["model"] == "stt-rt-v5"


@pytest.mark.asyncio
async def test_soniox_excludes_interim_tokens(
    stt_settings: Settings, audio_pcm_bytes: bytes
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
    provider = SonioxSTTProvider(stt_settings)

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
    assert result.complete_transcript == "hello world"
    assert result.word_count == 2


@pytest.mark.asyncio
async def test_soniox_keeps_space_tokens(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """A standalone " " token is a word boundary, not an empty token to skip."""
    events: list[Any] = [
        {"tokens": [{"text": "hello", "is_final": True}]},
        {"tokens": [{"text": " ", "is_final": True}]},
        {"tokens": [{"text": "world", "is_final": True}]},
        {"finished": True},
    ]
    provider = SonioxSTTProvider(stt_settings)

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
    assert result.complete_transcript == "hello world"
    assert result.word_count == 2
    assert result.partial_transcripts == []


# ---------------------------------------------------------------------------
# Provider name and model
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    assert make_provider().name == "soniox"


def test_provider_model() -> None:
    assert make_provider().model == "stt-rt-v5"


def test_stt_rt_v4_still_supported() -> None:
    assert SonioxSTTProvider(make_settings(), model="stt-rt-v4").model == "stt-rt-v4"


# ---------------------------------------------------------------------------
# Invalid construction
# ---------------------------------------------------------------------------


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="soniox_api_key is required"):
        SonioxSTTProvider(make_settings(soniox_api_key=None))


# ---------------------------------------------------------------------------
# Invalid sample rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soniox_wrong_sample_rate(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = SonioxSTTProvider(stt_settings)
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
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """The config hardcodes mono 16-bit, so non-matching input is rejected upfront."""
    provider = SonioxSTTProvider(stt_settings)
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
async def test_soniox_error_event(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("soniox", "events-error")
    provider = SonioxSTTProvider(stt_settings)

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
async def test_soniox_empty_stream(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = SonioxSTTProvider(stt_settings)

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


# ---------------------------------------------------------------------------
# Failure path — websocket transport exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soniox_connection_error(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """A connect failure surfaces as result.error, not a silent empty success."""
    provider = SonioxSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
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
async def test_soniox_surfaces_send_failure(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """A send failure inside the streaming task is surfaced, not swallowed by gather."""

    def _raise_on_audio(msg: object) -> None:
        if isinstance(msg, (bytes, bytearray)):
            raise RuntimeError("send boom")

    ws = FakeWebSocket([], on_send=_raise_on_audio)
    cm = fake_connect(ws)

    provider = SonioxSTTProvider(stt_settings)
    with patch("coval_bench.providers.stt._streaming.ws_client.connect", return_value=cm):
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
