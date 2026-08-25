# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ElevenLabs WebSocket TTS provider."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.elevenlabs import ElevenLabsTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_MODEL = "eleven_v3_conversational"
_FLASH_MODEL = "eleven_flash_v2_5"
_VOICE = "IKne3meq5aSn9XLyUdCD"


def _dialogue_events(pcm_chunks: list[bytes]) -> list[str | bytes]:
    events: list[str | bytes] = [
        json.dumps({"audio": base64.b64encode(chunk).decode()}) for chunk in pcm_chunks
    ]
    events.append(json.dumps({"is_final": True}))
    return events


def _stream_input_events(pcm_chunks: list[bytes]) -> list[str | bytes]:
    events: list[str | bytes] = [
        json.dumps({"audio": base64.b64encode(chunk).decode()}) for chunk in pcm_chunks
    ]
    events.append(json.dumps({"isFinal": True}))
    return events


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_happy_path(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_dialogue_events([make_pcm_bytes(240)]))
    provider = ElevenLabsTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.elevenlabs.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Eleven v3 Conversational")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None and result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "elevenlabs"
    assert result.model == _MODEL
    assert result.voice == _VOICE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_elevenlabs_url_and_frames(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_dialogue_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["headers"] = kwargs.get("additional_headers")
        return ws

    provider = ElevenLabsTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.elevenlabs.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    parts = urlsplit(str(captured["url"]))
    assert parts.scheme == "wss"
    assert parts.netloc == "api.elevenlabs.io"
    assert parts.path == "/v1/text-to-dialogue/stream-input"
    assert parse_qs(parts.query) == {
        "model_id": [_MODEL],
        "output_format": ["pcm_24000"],
    }
    assert captured["headers"] == {"xi-api-key": "test-elevenlabs-key"}
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent == [
        {"voices": [_VOICE]},
        {
            "inputs": [{"text": "Hello world", "voice_id": _VOICE}],
            "close_socket": True,
        },
    ]
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_elevenlabs_skips_empty_audio(fake_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"audio": ""}),
        json.dumps({"audio": base64.b64encode(make_pcm_bytes(240)).decode()}),
        json.dumps({"is_final": True}),
    ]
    ws = FakeWebSocket(events)
    provider = ElevenLabsTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.elevenlabs.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_elevenlabs_ttfa_on_first_chunk(fake_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_dialogue_events(chunks))
    provider = ElevenLabsTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    times = iter([0.0, 0.1])

    with (
        patch(
            "coval_bench.providers.tts.elevenlabs.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.elevenlabs.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Stream-input WebSocket (eleven_flash_v2_5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_flash_happy_path(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_stream_input_events([make_pcm_bytes(240)]))
    provider = ElevenLabsTTSProvider(fake_settings, model=_FLASH_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.elevenlabs.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Eleven Flash")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None and result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "elevenlabs"
    assert result.model == _FLASH_MODEL
    assert result.voice == _VOICE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_elevenlabs_flash_url_and_frames(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_stream_input_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["headers"] = kwargs.get("additional_headers")
        return ws

    provider = ElevenLabsTTSProvider(fake_settings, model=_FLASH_MODEL, voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.elevenlabs.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    parts = urlsplit(str(captured["url"]))
    assert parts.scheme == "wss"
    assert parts.netloc == "api.elevenlabs.io"
    assert parts.path == f"/v1/text-to-speech/{_VOICE}/stream-input"
    assert parse_qs(parts.query) == {
        "model_id": [_FLASH_MODEL],
        "output_format": ["pcm_24000"],
    }
    assert captured["headers"] == {"xi-api-key": "test-elevenlabs-key"}
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent == [
        {"text": " "},
        {"text": "Hello world "},
        {"text": ""},
    ]
    if result.audio_path is not None:
        result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_error_event(fake_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"error": "invalid_voice", "message": "voice not registered", "code": 1008})
    ]
    ws = FakeWebSocket(events)
    provider = ElevenLabsTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.elevenlabs.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "invalid_voice" in result.error
    assert "voice not registered" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_elevenlabs_connect_exception(fake_settings: Settings) -> None:
    provider = ElevenLabsTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.elevenlabs.ws_client.connect",
        side_effect=OSError("synthetic network failure"),
    ):
        result = await provider.synthesize("hi")

    assert result.error is not None
    assert "synthetic network failure" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_elevenlabs_no_audio(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_dialogue_events([]))
    provider = ElevenLabsTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.elevenlabs.ws_client.connect", return_value=ws):
        result = await provider.synthesize("silence")

    assert result.error is not None
    # The silent-failure error carries the last non-audio frames for diagnosis.
    assert "is_final" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_elevenlabs_skips_non_json_frames(fake_settings: Settings) -> None:
    events: list[str | bytes] = [
        "not json",
        json.dumps({"audio": base64.b64encode(make_pcm_bytes(240)).decode()}),
        json.dumps({"is_final": True}),
    ]
    ws = FakeWebSocket(events)
    provider = ElevenLabsTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.elevenlabs.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is not None
    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_elevenlabs_name_and_model(fake_settings: Settings) -> None:
    p = ElevenLabsTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)
    assert p.name == f"elevenlabs-{_MODEL}"
    assert p.model == _MODEL


def test_elevenlabs_rejects_unsupported_model(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unsupported ElevenLabs model"):
        ElevenLabsTTSProvider(fake_settings, model="eleven_turbo_v2_5", voice="v")


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_elevenlabs_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        elevenlabs_api_key=None,
    )
    with pytest.raises(ValueError, match="elevenlabs_api_key"):
        ElevenLabsTTSProvider(settings_no_key, model=_MODEL, voice="v")
