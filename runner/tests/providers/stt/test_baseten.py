# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.baseten (BasetenSTTProvider).

All tests use FakeWebSocket — no live network calls. The fixtures mirror the
Baseten Whisper wire protocol: ``transcription`` messages carry ``segments`` and
an ``is_final`` flag, two ``end_audio`` messages bracket the stream
(``acknowledged`` then ``finished``), and the receiver stops only on
``finished``. A small PCM buffer keeps the real-time 512-sample pacing fast.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.baseten import BasetenSTTProvider
from tests.providers.stt.conftest import FakeWebSocket

_WS_URL = "wss://model-test.api.baseten.co/environments/production/websocket"

# A few frames of PCM — enough to exercise framing/padding without the 3 s
# fixture's real-time pacing dominating the test runtime.
_SMALL_PCM = b"\x01\x02" * 1100  # 2200 bytes -> two 512-sample frames (last padded)


def make_provider(api_key: SecretStr | None = None) -> BasetenSTTProvider:
    return BasetenSTTProvider(api_key=api_key or SecretStr("test-key"), ws_url=_WS_URL)


def _fake_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_baseten_success(fake_api_key: SecretStr) -> None:
    events: list[Any] = [
        {"type": "transcription", "segments": [{"text": "hello"}], "is_final": False},
        {"type": "end_audio", "body": {"status": "acknowledged"}},
        {"type": "transcription", "segments": [{"text": "hello world"}], "is_final": True},
        {"type": "end_audio", "body": {"status": "finished"}},
    ]
    provider = BasetenSTTProvider(api_key=fake_api_key, ws_url=_WS_URL)

    with patch(
        "coval_bench.providers.stt.baseten.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.ttft_seconds is not None and result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert result.complete_transcript == "hello world"
    assert result.word_count == 2
    assert result.audio_to_final_seconds is not None
    assert result.partial_transcripts == ["hello"]
    wer = compute_wer("hello world", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_baseten_accumulates_multi_segment_finals(fake_api_key: SecretStr) -> None:
    """Multiple final messages (one per VAD segment) concatenate in order."""
    events: list[Any] = [
        {"type": "transcription", "segments": [{"text": "hello"}], "is_final": True},
        {"type": "transcription", "segments": [{"text": "world"}], "is_final": True},
        {"type": "end_audio", "body": {"status": "finished"}},
    ]
    provider = BasetenSTTProvider(api_key=fake_api_key, ws_url=_WS_URL)

    with patch(
        "coval_bench.providers.stt.baseten.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.complete_transcript == "hello world"
    assert result.word_count == 2


@pytest.mark.asyncio
async def test_baseten_ignores_acknowledged_end_audio(fake_api_key: SecretStr) -> None:
    """The 'acknowledged' end_audio must not stop the receive loop early."""
    events: list[Any] = [
        {"type": "end_audio", "body": {"status": "acknowledged"}},
        {"type": "transcription", "segments": [{"text": "after ack"}], "is_final": True},
        {"type": "end_audio", "body": {"status": "finished"}},
    ]
    provider = BasetenSTTProvider(api_key=fake_api_key, ws_url=_WS_URL)

    with patch(
        "coval_bench.providers.stt.baseten.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.complete_transcript == "after ack"


def test_provider_name() -> None:
    assert make_provider().name == "baseten"


def test_provider_model() -> None:
    assert make_provider().model == "whisper-large-v3"


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Baseten STT model"):
        BasetenSTTProvider(api_key=SecretStr("k"), model="whisper-tiny", ws_url=_WS_URL)


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="baseten_api_key is required"):
        BasetenSTTProvider(api_key=None, ws_url=_WS_URL)


def test_missing_ws_url_raises() -> None:
    with pytest.raises(ValueError, match="baseten_whisper_url is required"):
        BasetenSTTProvider(api_key=SecretStr("k"), ws_url=None)


@pytest.mark.asyncio
async def test_baseten_wrong_sample_rate(fake_api_key: SecretStr) -> None:
    provider = BasetenSTTProvider(api_key=fake_api_key, ws_url=_WS_URL)
    result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 8000)
    assert result.error is not None
    assert "16 kHz" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_baseten_rejects_non_mono(fake_api_key: SecretStr) -> None:
    provider = BasetenSTTProvider(api_key=fake_api_key, ws_url=_WS_URL)
    result = await provider.measure_ttft(_SMALL_PCM, 2, 2, 16000)
    assert result.error is not None
    assert "mono 16-bit" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_baseten_error_event(fake_api_key: SecretStr) -> None:
    events: list[Any] = [{"type": "error", "message": "Invalid or expired API key"}]
    provider = BasetenSTTProvider(api_key=fake_api_key, ws_url=_WS_URL)

    with patch(
        "coval_bench.providers.stt.baseten.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is not None
    assert "Invalid or expired API key" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_baseten_stream_ends_without_final(fake_api_key: SecretStr) -> None:
    """A stream that closes with no final transcript surfaces an error, not a silent pass."""
    provider = BasetenSTTProvider(api_key=fake_api_key, ws_url=_WS_URL)

    with patch(
        "coval_bench.providers.stt.baseten.ws_client.connect",
        return_value=_fake_connect([]),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is not None
    assert "before a final transcription" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_baseten_connection_error(fake_api_key: SecretStr) -> None:
    provider = BasetenSTTProvider(api_key=fake_api_key, ws_url=_WS_URL)

    with patch(
        "coval_bench.providers.stt.baseten.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.complete_transcript is None
