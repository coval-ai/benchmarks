# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Deepdub streaming WebSocket TTS provider."""

from __future__ import annotations

import base64
import json
import wave
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts.deepdub import SAMPLE_RATE, DeepdubTTSProvider
from tests.providers.fakes import FakeWebSocket

from .conftest import make_pcm_bytes

_MODEL = "dd-etts-3.3"
_VOICE = "f60bad7d-1667-42ff-8410-e60b71d0cc34"


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "deepdub_api_key": SecretStr("test-deepdub-key"),
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _welcome() -> str:
    return json.dumps({"action": "status", "connectionId": "conn-1", "message": "connected"})


def _ack() -> str:
    return json.dumps({"data": "", "generationId": "gen-1", "isFinished": False})


def _chunk(pcm: bytes, index: int, is_final: bool = False) -> str:
    return json.dumps(
        {
            "index": index,
            "generationId": "gen-1",
            "data": base64.b64encode(pcm).decode(),
            "isFinished": is_final,
            "isFinal": is_final,
        }
    )


def _events(pcm_chunks: list[bytes]) -> list[str | bytes]:
    events: list[str | bytes] = [_welcome(), _ack()]
    last = len(pcm_chunks) - 1
    events.extend(
        _chunk(pcm, index, is_final=index == last) for index, pcm in enumerate(pcm_chunks)
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
    assert captured["url"] == "wss://wss.deepdub.ai/ws"
    assert captured["additional_headers"] == {"x-api-key": "test-deepdub-key"}
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent == [
        {
            "action": "stream-config",
            "data": {
                "model": _MODEL,
                "voicePromptId": _VOICE,
                "locale": "en-US",
                "format": "s16le",
                "sampleRate": SAMPLE_RATE,
                "realtime": True,
            },
        },
        {"action": "stream-text", "data": {"text": "Hello world"}},
        {"action": "end-stream"},
    ]
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (
            {"action": "error", "message": "queue route not found for model: x", "time": 1},
            "deepdub session error: queue route not found for model: x",
        ),
        (
            {"generationId": "gen-1", "error": "Rate limit exceeded"},
            "deepdub worker error: Rate limit exceeded",
        ),
    ],
)
async def test_deepdub_tts_error_frame(
    deepdub_settings: Settings, frame: dict[str, object], expected: str
) -> None:
    ws = FakeWebSocket([_welcome(), json.dumps(frame)])
    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.deepdub.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error == expected
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_deepdub_tts_final_frame_audio_is_kept(deepdub_settings: Settings) -> None:
    """The isFinal frame may carry audio; it must land in the WAV before the break."""
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
async def test_deepdub_tts_close_before_isfinal_fails(deepdub_settings: Settings) -> None:
    """A generation can finish mid-turn; only isFinal ends it, so a close before is truncation."""
    finished_segment = json.dumps(
        {
            "index": 0,
            "generationId": "gen-1",
            "data": base64.b64encode(make_pcm_bytes(240)).decode(),
            "isFinished": True,
        }
    )
    ws = FakeWebSocket([_welcome(), finished_segment])
    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.deepdub.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "isFinal" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_deepdub_tts_silent_stream_is_a_failure(deepdub_settings: Settings) -> None:
    """A stream that finishes without audio or an error frame gets the stable reason."""
    events: list[str | bytes] = [
        _welcome(),
        json.dumps({"index": 0, "generationId": "gen-1", "isFinished": True, "isFinal": True}),
    ]
    ws = FakeWebSocket(events)
    provider = DeepdubTTSProvider(deepdub_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.deepdub.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert result.error.startswith("provider closed the stream without sending audio or an error")
    assert "isFinal" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


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
