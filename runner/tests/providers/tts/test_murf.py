# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Murf WebSocket streaming TTS provider."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.murf import MurfTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_VOICE = "Amara"


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        murfai_api_key="test-murf-key",
    )


def _events(pcm_chunks: list[bytes]) -> list[str | bytes]:
    events: list[str | bytes] = [
        json.dumps({"audio": base64.b64encode(chunk).decode()}) for chunk in pcm_chunks
    ]
    events.append(json.dumps({"final": True}))
    return events


@pytest.fixture()
def murf_settings() -> Settings:
    return _settings()


@pytest.mark.asyncio
async def test_murf_tts_happy_path(murf_settings: Settings) -> None:
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    provider = MurfTTSProvider(murf_settings, model="falcon-2", voice=_VOICE)

    with patch("coval_bench.providers.tts.murf.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Murf")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "murf"
    assert result.model == "falcon-2"
    assert result.voice == _VOICE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_murf_tts_url_and_frames(murf_settings: Settings) -> None:
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    captured: dict[str, str] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        return ws

    provider = MurfTTSProvider(murf_settings, model="falcon-2", voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.murf.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    parts = urlsplit(captured["url"])
    assert parts.scheme == "wss"
    assert parts.netloc == "global.api.murf.ai"
    assert parts.path == "/v1/speech/stream-input"
    assert parse_qs(parts.query) == {
        "api-key": ["test-murf-key"],
        "model": ["falcon-2"],
        "sample_rate": ["24000"],
        "channel_type": ["MONO"],
        "format": ["PCM"],
    }
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent == [
        {"voice_config": {"voiceId": _VOICE, "locale": "en-US"}},
        {"text": "Hello world", "end": True},
    ]
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_murf_tts_error_event(murf_settings: Settings) -> None:
    events: list[str | bytes] = [json.dumps({"error": "invalid voice"})]
    ws = FakeWebSocket(events)
    provider = MurfTTSProvider(murf_settings, model="falcon-2", voice=_VOICE)

    with patch("coval_bench.providers.tts.murf.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "invalid voice" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_murf_tts_skips_empty_audio(murf_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"audio": ""}),
        json.dumps({"audio": base64.b64encode(make_pcm_bytes(240)).decode()}),
        json.dumps({"final": True}),
    ]
    ws = FakeWebSocket(events)
    provider = MurfTTSProvider(murf_settings, model="falcon-2", voice=_VOICE)

    with patch("coval_bench.providers.tts.murf.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_murf_tts_ttfa_on_first_chunk(murf_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_events(chunks))
    provider = MurfTTSProvider(murf_settings, model="falcon-2", voice=_VOICE)

    times = iter([0.0, 0.1])

    with (
        patch(
            "coval_bench.providers.tts.murf.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.murf.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_murf_tts_invalid_model_raises(murf_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Murf TTS model"):
        MurfTTSProvider(murf_settings, model="gen2", voice=_VOICE)


def test_murf_tts_missing_voice_raises(murf_settings: Settings) -> None:
    with pytest.raises(ValueError, match="requires a voice"):
        MurfTTSProvider(murf_settings, model="falcon-2", voice="")


def test_murf_tts_missing_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
    )
    with pytest.raises(ValueError, match="murfai_api_key"):
        MurfTTSProvider(settings, model="falcon-2", voice=_VOICE)


def test_murf_tts_provider_name(murf_settings: Settings) -> None:
    provider = MurfTTSProvider(murf_settings, model="falcon-2", voice=_VOICE)
    assert provider.name == "murf-falcon-2"
    assert provider.model == "falcon-2"
