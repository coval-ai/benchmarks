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
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.baseten import BasetenSTTProvider, endpoint_url
from tests.providers.stt.conftest import FakeWebSocket

_WS_URL = "wss://model-test.api.baseten.co/environments/production/websocket"
_QWEN_MODEL = "qwen3-asr-1.7b"

# A few frames of PCM — enough to exercise framing/padding without the 3 s
# fixture's real-time pacing dominating the test runtime.
_SMALL_PCM = b"\x01\x02" * 1100  # 2200 bytes -> two 512-sample frames (last padded)


def make_provider(api_key: SecretStr | None = None) -> BasetenSTTProvider:
    return BasetenSTTProvider(api_key=api_key or SecretStr("test-key"), ws_url=_WS_URL)


def _fake_connect(events: list[Any], sent: list[Any] | None = None) -> Any:
    ws = FakeWebSocket(events, on_send=None if sent is None else sent.append)
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


# ---------------------------------------------------------------------------
# Qwen3-ASR: the input_audio_buffer protocol
# ---------------------------------------------------------------------------


def _qwen_provider(api_key: SecretStr) -> BasetenSTTProvider:
    return BasetenSTTProvider(api_key=api_key, model=_QWEN_MODEL, ws_url=_WS_URL)


@pytest.mark.asyncio
async def test_qwen_buffer_success(fake_api_key: SecretStr) -> None:
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
    provider = _qwen_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.baseten.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.complete_transcript == "hello world"
    assert result.partial_transcripts == ["hello"]
    assert result.audio_to_final_seconds is not None
    assert compute_wer("hello world", result.complete_transcript).wer_percentage == pytest.approx(
        0.0
    )


@pytest.mark.asyncio
async def test_qwen_flush_with_empty_segments_still_stops(fake_api_key: SecretStr) -> None:
    """A flush carrying no text must end the stream rather than hang on it."""
    events: list[Any] = [
        {"type": "transcription", "segments": [{"text": "done"}], "is_final": True},
        {"type": "transcription", "segments": [], "is_end_of_audio_flush": True},
    ]
    provider = _qwen_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.baseten.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert result.error is None
    assert result.complete_transcript == "done"


@pytest.mark.asyncio
async def test_qwen_sends_base64_appends_then_commit(fake_api_key: SecretStr) -> None:
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
    provider = _qwen_provider(fake_api_key)

    with patch(
        "coval_bench.providers.stt.baseten.ws_client.connect",
        return_value=_fake_connect(events, sent),
    ):
        await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    assert not any(isinstance(msg, (bytes, bytearray)) for msg in sent)
    frames = [json.loads(msg) for msg in sent]
    assert "streaming_params" in frames[0]
    appends = [f for f in frames if f.get("type") == "input_audio_buffer.append"]
    assert appends, "no audio was appended"
    reassembled = b"".join(base64.b64decode(f["audio"]) for f in appends)
    assert reassembled == _SMALL_PCM
    assert frames[-1] == {"type": "input_audio_buffer.commit"}


@pytest.mark.asyncio
async def test_whisper_still_sends_binary_frames_and_end_audio(fake_api_key: SecretStr) -> None:
    """Regression guard on the protocol split: Whisper must not get JSON audio."""
    events: list[Any] = [
        {"type": "transcription", "segments": [{"text": "ok"}], "is_final": True},
        {"type": "end_audio", "body": {"status": "finished"}},
    ]
    sent: list[Any] = []
    provider = BasetenSTTProvider(api_key=fake_api_key, ws_url=_WS_URL)

    with patch(
        "coval_bench.providers.stt.baseten.ws_client.connect",
        return_value=_fake_connect(events, sent),
    ):
        await provider.measure_ttft(_SMALL_PCM, 1, 2, 16000, 0.1)

    binary = [msg for msg in sent if isinstance(msg, (bytes, bytearray))]
    assert binary, "no binary audio frames were sent"
    assert {len(frame) for frame in binary} == {1024}
    assert json.loads(sent[-1]) == {"type": "end_audio"}


# ---------------------------------------------------------------------------
# Per-model endpoint resolution
# ---------------------------------------------------------------------------


def _url_settings(**kwargs: Any) -> Any:
    defaults = {
        "baseten_api_key": SecretStr("test-key"),
        "baseten_whisper_url": None,
        "baseten_qwen_asr_url": None,
    }
    return SimpleNamespace(**{**defaults, **kwargs})


def test_endpoint_url_is_per_model() -> None:
    settings = _url_settings(
        baseten_whisper_url="wss://whisper",
        baseten_qwen_asr_url="wss://qwen",
    )
    assert endpoint_url(settings, "whisper-large-v3") == "wss://whisper"
    assert endpoint_url(settings, _QWEN_MODEL) == "wss://qwen"


def test_endpoint_url_unknown_model_is_none() -> None:
    assert endpoint_url(_url_settings(), "whisper-tiny") is None


def test_missing_url_error_names_the_models_setting() -> None:
    with pytest.raises(ValueError, match="baseten_qwen_asr_url is required"):
        BasetenSTTProvider(api_key=SecretStr("k"), model=_QWEN_MODEL, ws_url=None)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_hits_every_configured_endpoint() -> None:
    urls: list[str] = []

    def _connect(url: str, **_: Any) -> Any:
        urls.append(url)
        return _fake_connect([{"type": "end_audio", "body": {"status": "finished"}}])

    settings = _url_settings(baseten_whisper_url="wss://v1", baseten_qwen_asr_url="wss://qwen")

    with (
        patch("coval_bench.providers.stt.baseten._WARMUP_SECONDS", 0.02),
        patch("coval_bench.providers.stt.baseten.ws_client.connect", side_effect=_connect),
    ):
        await BasetenSTTProvider.warmup(settings)

    # v2 is unconfigured here, so it must not be dialled.
    assert sorted(urls) == ["wss://qwen", "wss://v1"]


@pytest.mark.asyncio
async def test_warmup_without_api_key_is_a_noop() -> None:
    settings = _url_settings(baseten_api_key=None, baseten_whisper_url="wss://v1")

    with patch("coval_bench.providers.stt.baseten.ws_client.connect") as connect:
        await BasetenSTTProvider.warmup(settings)

    connect.assert_not_called()


@pytest.mark.asyncio
async def test_warmup_survives_a_dead_endpoint() -> None:
    """One endpoint failing must not stop the others from being warmed."""
    settings = _url_settings(baseten_whisper_url="wss://v1", baseten_qwen_asr_url="wss://qwen")
    seen: list[str] = []

    def _connect(url: str, **_: Any) -> Any:
        seen.append(url)
        if url == "wss://v1":
            raise OSError("connection refused")
        return _fake_connect([{"type": "end_audio", "body": {"status": "finished"}}])

    with (
        patch("coval_bench.providers.stt.baseten._WARMUP_SECONDS", 0.02),
        patch("coval_bench.providers.stt.baseten.ws_client.connect", side_effect=_connect),
    ):
        await BasetenSTTProvider.warmup(settings)

    assert sorted(seen) == ["wss://qwen", "wss://v1"]


@pytest.mark.asyncio
async def test_warmup_treats_no_final_as_success() -> None:
    """The tone clip has no speech, so a close without a final is a warm endpoint."""
    settings = _url_settings(baseten_whisper_url="wss://whisper")

    def _connect(url: str, **_: Any) -> Any:
        return _fake_connect([{"type": "end_audio", "body": {"status": "finished"}}])

    with (
        patch("coval_bench.providers.stt.baseten._WARMUP_SECONDS", 0.02),
        patch("coval_bench.providers.stt.baseten.ws_client.connect", side_effect=_connect),
        patch("coval_bench.providers.stt.baseten.logger") as log,
    ):
        await BasetenSTTProvider.warmup(settings)

    prewarms = [c.kwargs for c in log.info.call_args_list if c.args[0] == "baseten_stt_prewarm"]
    assert prewarms and prewarms[0]["error"] is None


@pytest.mark.asyncio
async def test_warmup_still_reports_real_errors() -> None:
    settings = _url_settings(baseten_whisper_url="wss://whisper")
    events: list[Any] = [{"type": "error", "message": "Invalid or expired API key"}]

    with (
        patch("coval_bench.providers.stt.baseten._WARMUP_SECONDS", 0.02),
        patch(
            "coval_bench.providers.stt.baseten.ws_client.connect",
            side_effect=lambda url, **_: _fake_connect(events),
        ),
        patch("coval_bench.providers.stt.baseten.logger") as log,
    ):
        await BasetenSTTProvider.warmup(settings)

    prewarms = [c.kwargs for c in log.info.call_args_list if c.args[0] == "baseten_stt_prewarm"]
    assert prewarms and "Invalid or expired API key" in prewarms[0]["error"]
