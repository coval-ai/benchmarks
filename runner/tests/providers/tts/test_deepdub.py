# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Deepdub WebSocket TTS provider."""

from __future__ import annotations

import base64
import json
import wave
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts.deepdub import SAMPLE_RATE, DeepdubTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_MODEL = "dd-etts-3.3"
_VOICE = "02215cf5-04af-46f3-a061-48a4c81989bf"


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql://runner:password@localhost:5432/benchmarks",
        "dataset_bucket": "test-bucket",
        "dataset_id": "stt-v1",
        "runner_sha": "test",
        "log_level": "DEBUG",
        "deepdub_api_key": SecretStr("test-deepdub-key"),
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _ack() -> str:
    return json.dumps({"data": "", "generationId": "gen-1", "isFinished": False})


def _chunk(pcm: bytes, index: int, is_finished: bool = False) -> str:
    return json.dumps(
        {
            "index": index,
            "generationId": "gen-1",
            "data": base64.b64encode(pcm).decode(),
            "isFinished": is_finished,
        }
    )


def _events(pcm_chunks: list[bytes]) -> list[str | bytes]:
    events: list[str | bytes] = [_ack()]
    last = len(pcm_chunks) - 1
    events.extend(
        _chunk(pcm, index, is_finished=index == last) for index, pcm in enumerate(pcm_chunks)
    )
    return events


@pytest.fixture()
def deepdub_settings() -> Settings:
    return _settings()


@pytest.mark.asyncio
async def test_deepdub_tts_happy_path(deepdub_settings: Settings) -> None:
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.deepdub.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Deepdub")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "deepdub"
    assert result.model == _MODEL
    assert result.voice == _VOICE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_deepdub_tts_url_headers_and_request_frame(deepdub_settings: Settings) -> None:
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured.update(kwargs)
        return ws

    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.deepdub.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert captured["url"] == "wss://wsapi.deepdub.ai/open"
    assert captured["additional_headers"] == {"x-api-key": "test-deepdub-key"}
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent == [
        {
            "action": "text-to-speech",
            "model": _MODEL,
            "targetText": "Hello world",
            "locale": "en-US",
            "voicePromptId": _VOICE,
            "format": "s16le",
            "sampleRate": SAMPLE_RATE,
            "realtime": True,
        }
    ]
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_deepdub_tts_error_frame(deepdub_settings: Settings) -> None:
    events: list[str | bytes] = [
        _ack(),
        json.dumps(
            {"error": "Rate limit exceeded", "errorType": "RateLimit", "generationId": "gen-1"}
        ),
    ]
    ws = FakeWebSocket(events)
    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.deepdub.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "RateLimit" in result.error
    assert "Rate limit exceeded" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_deepdub_tts_final_frame_audio_is_kept(deepdub_settings: Settings) -> None:
    """The isFinished frame may carry audio; it must land in the WAV before the break."""
    ws = FakeWebSocket(_events([make_pcm_bytes(240), make_pcm_bytes(240)]))
    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.deepdub.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is not None
    with wave.open(str(result.audio_path), "rb") as wav_file:
        assert wav_file.getnframes() == 480
        assert wav_file.getframerate() == SAMPLE_RATE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_deepdub_tts_ttfa_skips_empty_ack(deepdub_settings: Settings) -> None:
    """The empty-data ack must not start the audio clock; only real PCM does."""
    ws = FakeWebSocket(_events([make_pcm_bytes(240), make_pcm_bytes(240)]))
    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)

    times = iter([0.0, 0.1])

    with (
        patch(
            "coval_bench.providers.tts.deepdub.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.deepdub.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_deepdub_tts_close_before_isfinished_fails(deepdub_settings: Settings) -> None:
    """A clean close before the isFinished frame is truncation, never a scored result."""
    events: list[str | bytes] = [_ack(), _chunk(make_pcm_bytes(240), 0)]
    ws = FakeWebSocket(events)
    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.deepdub.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "isFinished" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_deepdub_tts_silent_stream_is_a_failure(deepdub_settings: Settings) -> None:
    """A stream that finishes without audio or an error frame gets the stable reason."""
    events: list[str | bytes] = [
        _ack(),
        json.dumps({"index": 0, "generationId": "gen-1", "data": "", "isFinished": True}),
    ]
    ws = FakeWebSocket(events)
    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.deepdub.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert result.error.startswith("provider closed the stream without sending audio or an error")
    assert "isFinished" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


def test_deepdub_tts_invalid_model_raises(deepdub_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unsupported Deepdub model"):
        DeepdubTTSProvider(deepdub_settings, model="not-a-model", voice=_VOICE)


def test_deepdub_tts_missing_voice_raises(deepdub_settings: Settings) -> None:
    with pytest.raises(ValueError, match="requires a voice"):
        DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice="")


def test_deepdub_tts_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="deepdub_api_key"):
        DeepdubTTSProvider(_settings(deepdub_api_key=None), model=_MODEL, voice=_VOICE)


def test_deepdub_tts_provider_name(deepdub_settings: Settings) -> None:
    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)
    assert provider.name == "deepdub-dd-etts-3.3"
    assert provider.model == _MODEL
