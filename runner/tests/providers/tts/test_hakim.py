# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hakim AI WebSocket streaming TTS provider."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.hakim import HakimTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_MODEL = "hakim-fast-v1"
_VOICE = "amelia-en-us"


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        hakimai_api_key="test-hakim-key",
    )


def _events(pcm_chunks: list[bytes]) -> list[str | bytes]:
    events: list[str | bytes] = [
        json.dumps({"type": "session.created", "session_id": "sess_1"}),
        json.dumps({"type": "speech.started", "sample_rate": 24000, "channels": 1}),
    ]
    events.extend(pcm_chunks)
    events.append(json.dumps({"type": "speech.done", "usage": {"units": 11}}))
    return events


@pytest.fixture()
def hakim_settings() -> Settings:
    return _settings()


@pytest.mark.asyncio
async def test_hakim_tts_happy_path(hakim_settings: Settings) -> None:
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    provider = HakimTTSProvider(hakim_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.hakim.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Hakim")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "hakim"
    assert result.model == _MODEL
    assert result.voice == _VOICE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_hakim_tts_url_auth_and_frames(hakim_settings: Settings) -> None:
    ws = FakeWebSocket(_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["headers"] = kwargs.get("additional_headers")
        return ws

    provider = HakimTTSProvider(hakim_settings, model=_MODEL, voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.hakim.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert captured["url"] == "wss://api.tryhakim.ai/v1/audio/speech/stream"
    assert captured["headers"] == {"Authorization": "Bearer test-hakim-key"}
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent == [
        {
            "type": "speech.create",
            "input": "Hello world",
            "model": _MODEL,
            "voice": _VOICE,
            "cfg": 3,
        }
    ]
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_hakim_tts_error_frame(hakim_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"type": "session.created", "session_id": "sess_1"}),
        json.dumps({"type": "error", "code": "voice_not_found", "message": "no such voice"}),
    ]
    ws = FakeWebSocket(events)
    provider = HakimTTSProvider(hakim_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.hakim.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "voice_not_found" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_hakim_tts_truncated_stream_is_error(hakim_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"type": "session.created", "session_id": "sess_1"}),
        json.dumps({"type": "speech.started", "sample_rate": 24000, "channels": 1}),
        make_pcm_bytes(240),
    ]
    ws = FakeWebSocket(events)
    provider = HakimTTSProvider(hakim_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.hakim.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "speech.done" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_hakim_tts_ignores_lifecycle_frames(hakim_settings: Settings) -> None:
    events: list[str | bytes] = [
        json.dumps({"type": "session.created", "session_id": "sess_1"}),
        json.dumps({"type": "speech.started", "sample_rate": 24000, "channels": 1}),
        make_pcm_bytes(240),
        json.dumps({"type": "session.usage", "session_characters": 5}),
        make_pcm_bytes(240),
        json.dumps({"type": "speech.done", "usage": {"units": 5}}),
    ]
    ws = FakeWebSocket(events)
    provider = HakimTTSProvider(hakim_settings, model=_MODEL, voice=_VOICE)

    with patch("coval_bench.providers.tts.hakim.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_hakim_tts_ttfa_on_first_binary_frame(hakim_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_events(chunks))
    provider = HakimTTSProvider(hakim_settings, model=_MODEL, voice=_VOICE)

    times = iter([0.0, 0.1])

    with (
        patch(
            "coval_bench.providers.tts.hakim.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.hakim.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_hakim_tts_invalid_model_raises(hakim_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Hakim TTS model"):
        HakimTTSProvider(hakim_settings, model="hakim-v2", voice=_VOICE)


def test_hakim_tts_missing_voice_raises(hakim_settings: Settings) -> None:
    with pytest.raises(ValueError, match="requires a voice"):
        HakimTTSProvider(hakim_settings, model=_MODEL, voice="")


def test_hakim_tts_missing_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
    )
    with pytest.raises(ValueError, match="hakimai_api_key"):
        HakimTTSProvider(settings, model=_MODEL, voice=_VOICE)


def test_hakim_tts_empty_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        hakimai_api_key="",
    )
    with pytest.raises(ValueError, match="hakimai_api_key"):
        HakimTTSProvider(settings, model=_MODEL, voice=_VOICE)


def test_hakim_tts_provider_name(hakim_settings: Settings) -> None:
    provider = HakimTTSProvider(hakim_settings, model=_MODEL, voice=_VOICE)
    assert provider.name == "hakim-hakim-fast-v1"
    assert provider.model == _MODEL
