# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.zoom (ZoomSTTProvider).

All tests use FakeWebSocket — no live network calls are made. Events mirror
the Scribe live wire protocol: interim ``transcription.delta``, turn-final
``transcription.completed``, ``session.closed`` after ``session.close``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import jwt
import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.stt.zoom import ZoomSTTProvider
from tests.providers.stt.conftest import fake_connect, make_settings

_TEST_SECRET = "test-secret-zoom-0000000000000000"  # noqa: S105 - not a real credential


_SUCCESS_EVENTS: list[Any] = [
    {
        "type": "transcription.completed",
        "item_id": "item-1",
        "transcript": "hello world",
        "audio_start_ms": 0,
        "audio_end_ms": 1400,
        "transcription_latency_ms": 180,
    },
    {
        "type": "transcription.completed",
        "item_id": "item-2",
        "transcript": "how are you",
        "audio_start_ms": 1400,
        "audio_end_ms": 2900,
        "transcription_latency_ms": 165,
    },
    {"type": "session.closed"},
]


@pytest.mark.asyncio
async def test_zoom_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    sent: list[Any] = []
    provider = ZoomSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(_SUCCESS_EVENTS, sent),
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
    assert result.first_token_content == "hello world"  # noqa: S105 - transcript fixture text
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    assert result.audio_to_final_seconds is not None

    assert json.loads(sent[0]) == {
        "type": "session.update",
        "config": {"language": "en-US", "model": "zoom-scribe-en"},
        "audio": {"format": "pcm16"},
    }
    assert all(isinstance(frame, bytes) for frame in sent[1:-1])
    assert json.loads(sent[-1]) == {"type": "session.close"}


@pytest.mark.asyncio
async def test_zoom_ignores_delta_events(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """Deltas never arrive under the working config; if one does, it must not
    count as a first token — a set TTFT would mask a missing-final failure."""
    events: list[Any] = [
        {"type": "transcription.delta", "delta": "wrong guess"},
        {"type": "transcription.completed", "transcript": "hello world"},
        {"type": "session.closed"},
    ]
    provider = ZoomSTTProvider(stt_settings)

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
    assert result.partial_transcripts == []
    assert result.first_token_content == "hello world"  # noqa: S105 - transcript fixture text


@pytest.mark.asyncio
async def test_zoom_non_fatal_error_does_not_end_the_run(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [
        {"type": "error", "error": {"code": "some_hint", "message": "heads up", "fatal": False}},
        {"type": "transcription.completed", "transcript": "hello world"},
        {"type": "session.closed"},
    ]
    provider = ZoomSTTProvider(stt_settings)

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


@pytest.mark.asyncio
async def test_zoom_fatal_error_stops_the_send(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """A fatal error aborts the stream, so the sender stops instead of pacing
    out the rest of the clip; the fatal message survives the send failure."""
    sent: list[Any] = []
    events: list[Any] = [
        {"type": "error", "error": {"code": "unauthorized", "message": "bad token", "fatal": True}}
    ]
    provider = ZoomSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events, sent),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error == "bad token"
    assert result.complete_transcript is None
    assert json.dumps({"type": "session.close"}) not in sent


def test_zoom_requires_both_credentials() -> None:
    with pytest.raises(ValueError, match="zoom_api_key is required"):
        ZoomSTTProvider(make_settings(zoom_api_key=None))
    with pytest.raises(ValueError, match="zoom_api_secret is required"):
        ZoomSTTProvider(make_settings(zoom_api_secret=None))


@pytest.mark.asyncio
async def test_zoom_rejects_non_16k_mono_pcm16(stt_settings: Settings) -> None:
    provider = ZoomSTTProvider(stt_settings)

    result = await provider.measure_ttft(
        audio_data=b"\x00\x00", channels=1, sample_width=2, sample_rate=8000
    )
    assert result.error is not None and "16 kHz" in result.error

    result = await provider.measure_ttft(
        audio_data=b"\x00\x00", channels=2, sample_width=2, sample_rate=16000
    )
    assert result.error is not None and "mono" in result.error


def test_zoom_jwt_claims() -> None:
    settings = make_settings(zoom_api_secret=SecretStr(_TEST_SECRET))
    provider = ZoomSTTProvider(settings)
    assert settings.zoom_api_key is not None

    claims = jwt.decode(
        provider._auth_token(),
        _TEST_SECRET,
        algorithms=["HS256"],
        issuer=settings.zoom_api_key.get_secret_value(),
    )
    assert claims["exp"] - claims["iat"] == 7200


@pytest.mark.asyncio
async def test_zoom_pro_row_selects_the_pro_model(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    sent: list[Any] = []
    provider = ZoomSTTProvider(stt_settings, model="scribe-pro")

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(_SUCCESS_EVENTS, sent),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert json.loads(sent[0])["config"] == {"language": "en-US", "model": "zoom-scribe-en-pro"}
