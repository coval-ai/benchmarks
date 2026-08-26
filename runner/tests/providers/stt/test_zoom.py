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
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.zoom import ZoomSTTProvider
from tests.providers.stt.conftest import FakeWebSocket

_TEST_SECRET = "test-secret-zoom-0000000000000000"  # noqa: S105 - not a real credential


def make_provider(api_key: SecretStr) -> ZoomSTTProvider:
    return ZoomSTTProvider(api_key=api_key, api_secret=SecretStr(_TEST_SECRET))


def _fake_connect(events: list[Any], sent: list[Any] | None = None) -> Any:
    ws = FakeWebSocket(events, on_send=sent.append if sent is not None else None)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


_SUCCESS_EVENTS: list[Any] = [
    {"type": "transcription.delta", "delta": "hello"},
    {"type": "transcription.delta", "delta": "hello wor"},
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
async def test_zoom_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    sent: list[Any] = []
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.zoom.ws_client.connect",
        return_value=_fake_connect(_SUCCESS_EVENTS, sent),
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
    assert result.first_token_content == "hello"  # noqa: S105 - transcript fixture text
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    assert result.audio_to_final_seconds is not None

    assert json.loads(sent[0])["type"] == "session.update"
    assert all(isinstance(frame, bytes) for frame in sent[1:-1])
    assert json.loads(sent[-1]) == {"type": "session.close"}


@pytest.mark.asyncio
async def test_zoom_deltas_stay_out_of_the_transcript(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [
        {"type": "transcription.delta", "delta": "wrong guess"},
        {"type": "transcription.completed", "transcript": "hello world"},
        {"type": "session.closed"},
    ]
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.zoom.ws_client.connect",
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
    assert result.partial_transcripts == ["wrong guess"]
    assert result.first_token_content == "wrong guess"  # noqa: S105 - transcript fixture text


@pytest.mark.asyncio
async def test_zoom_non_fatal_error_does_not_end_the_run(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [
        {"type": "error", "error": {"code": "some_hint", "message": "heads up", "fatal": False}},
        {"type": "transcription.completed", "transcript": "hello world"},
        {"type": "session.closed"},
    ]
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.zoom.ws_client.connect",
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


@pytest.mark.asyncio
async def test_zoom_fatal_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [
        {"type": "error", "error": {"code": "unauthorized", "message": "bad token", "fatal": True}}
    ]
    provider = make_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.zoom.ws_client.connect",
        return_value=_fake_connect(events),
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


def test_zoom_requires_both_credentials(fake_api_key: SecretStr) -> None:
    with pytest.raises(ValueError, match="zoom_api_key"):
        ZoomSTTProvider(api_key=None, api_secret=SecretStr("s"))
    with pytest.raises(ValueError, match="zoom_api_secret"):
        ZoomSTTProvider(api_key=fake_api_key, api_secret=None)
    with pytest.raises(ValueError, match="Invalid Zoom STT model"):
        ZoomSTTProvider(api_key=fake_api_key, api_secret=SecretStr("s"), model="scribe-9000")


@pytest.mark.asyncio
async def test_zoom_rejects_non_16k_mono_pcm16(fake_api_key: SecretStr) -> None:
    provider = make_provider(fake_api_key)

    result = await provider.measure_ttft(
        audio_data=b"\x00\x00", channels=1, sample_width=2, sample_rate=8000
    )
    assert result.error is not None and "16 kHz" in result.error

    result = await provider.measure_ttft(
        audio_data=b"\x00\x00", channels=2, sample_width=2, sample_rate=16000
    )
    assert result.error is not None and "mono" in result.error


def test_zoom_jwt_claims(fake_api_key: SecretStr) -> None:
    provider = make_provider(fake_api_key)

    claims = jwt.decode(
        provider._auth_token(),
        _TEST_SECRET,
        algorithms=["HS256"],
        issuer=fake_api_key.get_secret_value(),
    )
    assert claims["exp"] - claims["iat"] == 7200
