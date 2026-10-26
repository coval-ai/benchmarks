# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.baseten (BasetenSTTProvider).

All tests use FakeWebSocket — no live network calls. Two wire protocols are
covered. Whisper endpoints (``binary``) take raw 512-sample frames and are
bracketed by two ``end_audio`` messages (``acknowledged`` then ``finished``),
and the receiver stops only on ``finished``. The Qwen3-ASR endpoint
(``buffer``) takes base64 PCM in ``input_audio_buffer.append`` messages,
finalizes with ``input_audio_buffer.commit``, and ends on a transcription
flagged ``is_end_of_audio_flush``. Both share the ``transcription`` message
shape. A small PCM buffer keeps the real-time pacing fast.
"""

from __future__ import annotations

import base64
import json
from collections.abc import MutableMapping
from typing import Any
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

from coval_bench.config import Settings
from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.baseten import (
    _OPEN_TIMEOUT_S,
    _WARMUP_TIMEOUT_S,
    BasetenSTTProvider,
    endpoint_url,
)
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import make_settings

_QWEN_MODEL = "qwen3-asr-1.7b"

# A few frames of PCM — enough to exercise framing/padding without the 3 s
# fixture's real-time pacing dominating the test runtime.
_SMALL_PCM = b"\x01\x02" * 1100  # 2200 bytes -> two 512-sample frames (last padded)


@pytest.mark.asyncio
async def test_baseten_success(stt_settings: Settings) -> None:
    events: list[Any] = [
        {"type": "transcription", "segments": [{"text": "hello"}], "is_final": False},
        {"type": "end_audio", "body": {"status": "acknowledged"}},
        {"type": "transcription", "segments": [{"text": "hello world"}], "is_final": True},
        {"type": "end_audio", "body": {"status": "finished"}},
    ]
    provider = BasetenSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
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
async def test_baseten_accumulates_multi_segment_finals(stt_settings: Settings) -> None:
    """Multiple final messages (one per VAD segment) concatenate in order."""
    events: list[Any] = [
        {"type": "transcription", "segments": [{"text": "hello"}], "is_final": True},
        {"type": "transcription", "segments": [{"text": "world"}], "is_final": True},
        {"type": "end_audio", "body": {"status": "finished"}},
    ]
    provider = BasetenSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.complete_transcript == "hello world"
    assert result.word_count == 2


@pytest.mark.asyncio
async def test_baseten_ignores_acknowledged_end_audio(stt_settings: Settings) -> None:
    """The 'acknowledged' end_audio must not stop the receive loop early."""
    events: list[Any] = [
        {"type": "end_audio", "body": {"status": "acknowledged"}},
        {"type": "transcription", "segments": [{"text": "after ack"}], "is_final": True},
        {"type": "end_audio", "body": {"status": "finished"}},
    ]
    provider = BasetenSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.complete_transcript == "after ack"


def test_provider_name(stt_settings: Settings) -> None:
    assert BasetenSTTProvider(stt_settings).name == "baseten"


def test_provider_model(stt_settings: Settings) -> None:
    assert BasetenSTTProvider(stt_settings).model == "whisper-large-v3"


def test_invalid_model_raises(stt_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Baseten STT model"):
        BasetenSTTProvider(stt_settings, model="whisper-tiny")


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="baseten_api_key is required"):
        BasetenSTTProvider(make_settings(baseten_api_key=None))


def test_missing_ws_url_raises() -> None:
    with pytest.raises(ValueError, match="baseten_whisper_url is required"):
        BasetenSTTProvider(make_settings(baseten_whisper_url=None))


@pytest.mark.asyncio
async def test_baseten_wrong_sample_rate(stt_settings: Settings) -> None:
    provider = BasetenSTTProvider(stt_settings)
    result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 8000)
    assert result.error is not None
    assert "16 kHz" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_baseten_rejects_non_mono(stt_settings: Settings) -> None:
    provider = BasetenSTTProvider(stt_settings)
    result = await provider.measure_ttft(_SMALL_PCM, 2, 2, 16000)
    assert result.error is not None
    assert "mono 16-bit" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_baseten_error_event(stt_settings: Settings) -> None:
    events: list[Any] = [{"type": "error", "message": "Invalid or expired API key"}]
    provider = BasetenSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is not None
    assert "Invalid or expired API key" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_baseten_stream_ends_without_final(stt_settings: Settings) -> None:
    """A stream that closes with no final transcript surfaces an error, not a silent pass."""
    provider = BasetenSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect([]),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is not None
    assert "before a final transcription" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_baseten_connection_error(stt_settings: Settings) -> None:
    provider = BasetenSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.complete_transcript is None


# ---------------------------------------------------------------------------
# Qwen3-ASR: the input_audio_buffer protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qwen_buffer_success(stt_settings: Settings) -> None:
    """The buffer protocol ends on is_end_of_audio_flush, never on end_audio."""
    events: list[Any] = [
        {"type": "transcription", "segments": [{"text": "hello"}], "is_final": False},
        {
            "type": "transcription",
            "segments": [{"text": "hello world"}],
            "is_final": True,
            "is_end_of_audio_flush": True,
        },
    ]
    provider = BasetenSTTProvider(stt_settings, model=_QWEN_MODEL)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.complete_transcript == "hello world"
    assert result.partial_transcripts == ["hello"]
    assert result.audio_to_final_seconds is not None


@pytest.mark.asyncio
async def test_qwen_flush_with_empty_segments_still_stops(stt_settings: Settings) -> None:
    """A flush carrying no text must end the stream rather than hang on it."""
    events: list[Any] = [
        {"type": "transcription", "segments": [{"text": "done"}], "is_final": True},
        {"type": "transcription", "segments": [], "is_end_of_audio_flush": True},
    ]
    provider = BasetenSTTProvider(stt_settings, model=_QWEN_MODEL)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.complete_transcript == "done"


@pytest.mark.asyncio
async def test_qwen_sends_base64_appends_then_commit(stt_settings: Settings) -> None:
    """Audio goes out base64 in JSON, never as binary frames, and commit finalizes."""
    events: list[Any] = [
        {
            "type": "transcription",
            "segments": [{"text": "ok"}],
            "is_final": True,
            "is_end_of_audio_flush": True,
        },
    ]
    sent: list[Any] = []
    provider = BasetenSTTProvider(stt_settings, model=_QWEN_MODEL)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events, sent),
    ):
        await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert not any(isinstance(msg, (bytes, bytearray)) for msg in sent)
    frames = [json.loads(msg) for msg in sent]
    assert "streaming_params" in frames[0]
    appends = [f for f in frames if f.get("type") == "input_audio_buffer.append"]
    reassembled = b"".join(base64.b64decode(f["audio"]) for f in appends)
    assert reassembled == _SMALL_PCM
    assert frames[-1] == {"type": "input_audio_buffer.commit"}


@pytest.mark.asyncio
async def test_whisper_still_sends_binary_frames_and_end_audio(stt_settings: Settings) -> None:
    """Protocol-split regression guard: whisper keeps binary frames and the 45 s handshake."""
    events: list[Any] = [
        {"type": "transcription", "segments": [{"text": "ok"}], "is_final": True},
        {"type": "end_audio", "body": {"status": "finished"}},
    ]
    sent: list[Any] = []
    timeouts: list[float] = []

    def _connect(url: str, **kwargs: Any) -> Any:
        timeouts.append(kwargs["open_timeout"])
        return fake_connect(events, sent)

    provider = BasetenSTTProvider(stt_settings)
    with patch("coval_bench.providers.stt._streaming.ws_client.connect", side_effect=_connect):
        await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    binary = [msg for msg in sent if isinstance(msg, (bytes, bytearray))]
    assert {len(frame) for frame in binary} == {1024}
    assert json.loads(sent[-1]) == {"type": "end_audio"}
    assert timeouts == [_OPEN_TIMEOUT_S]


# ---------------------------------------------------------------------------
# Endpoint resolution and warmup
# ---------------------------------------------------------------------------


def _url_settings() -> Settings:
    return make_settings(baseten_whisper_url="wss://whisper", baseten_qwen_asr_url="wss://qwen")


def test_endpoint_url_resolution() -> None:
    settings = _url_settings()
    assert endpoint_url(settings, "whisper-large-v3") == "wss://whisper"
    assert endpoint_url(settings, _QWEN_MODEL) == "wss://qwen"
    assert endpoint_url(settings, "whisper-tiny") is None


def _prewarm_errors(logs: list[MutableMapping[str, Any]]) -> dict[str, Any]:
    return {log["model"]: log["error"] for log in logs if log["event"] == "baseten_stt_prewarm"}


@pytest.mark.asyncio
async def test_warmup_warms_every_configured_endpoint() -> None:
    """Dials each configured URL with the boot-length handshake; tone no-final is success."""
    calls: list[tuple[str, float]] = []

    def _connect(url: str, **kwargs: Any) -> Any:
        calls.append((url, kwargs["open_timeout"]))
        return fake_connect([{"type": "end_audio", "body": {"status": "finished"}}])

    with (
        patch("coval_bench.providers.stt.baseten._WARMUP_SECONDS", 0.02),
        patch("coval_bench.providers.stt._streaming.ws_client.connect", side_effect=_connect),
        capture_logs() as logs,
    ):
        await BasetenSTTProvider.warmup(_url_settings())

    assert sorted(calls) == [
        ("wss://qwen", _WARMUP_TIMEOUT_S),
        ("wss://whisper", _WARMUP_TIMEOUT_S),
    ]
    assert _prewarm_errors(logs) == {"whisper-large-v3": None, _QWEN_MODEL: None}


@pytest.mark.asyncio
async def test_warmup_survives_a_dead_endpoint() -> None:
    """One endpoint failing must not stop the others, and its error must surface."""
    seen: list[str] = []

    def _connect(url: str, **_: Any) -> Any:
        seen.append(url)
        if url == "wss://whisper":
            raise OSError("connection refused")
        return fake_connect([{"type": "end_audio", "body": {"status": "finished"}}])

    with (
        patch("coval_bench.providers.stt.baseten._WARMUP_SECONDS", 0.02),
        patch("coval_bench.providers.stt._streaming.ws_client.connect", side_effect=_connect),
        capture_logs() as logs,
    ):
        await BasetenSTTProvider.warmup(_url_settings())

    assert sorted(seen) == ["wss://qwen", "wss://whisper"]
    errors = _prewarm_errors(logs)
    assert errors[_QWEN_MODEL] is None
    assert "connection refused" in errors["whisper-large-v3"]
