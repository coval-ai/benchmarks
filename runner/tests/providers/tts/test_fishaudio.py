# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Fish Audio WebSocket TTS provider."""

from __future__ import annotations

from unittest.mock import patch

import ormsgpack
import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts.fishaudio import FishAudioTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_VOICE = "802e3bc2b27e49c2995d23ef70e6ac89"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql://runner:password@localhost:5432/benchmarks",
        "dataset_bucket": "test-bucket",
        "dataset_id": "stt-v1",
        "runner_sha": "test",
        "log_level": "DEBUG",
        "fishaudio_api_key": SecretStr("test-fishaudio-key"),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _finish_events(pcm_chunks: list[bytes]) -> list[bytes | str]:
    events: list[bytes | str] = [
        ormsgpack.packb({"event": "audio", "audio": chunk}) for chunk in pcm_chunks
    ]
    events.append(ormsgpack.packb({"event": "finish", "reason": "stop"}))
    return events


@pytest.fixture()
def fishaudio_settings() -> Settings:
    return _settings()


@pytest.mark.asyncio
async def test_fishaudio_tts_happy_path(fishaudio_settings: Settings) -> None:
    ws = FakeWebSocket(_finish_events([make_pcm_bytes(240)]))
    provider = FishAudioTTSProvider(fishaudio_settings, model="s1", voice=_VOICE)

    with patch("coval_bench.providers.tts.fishaudio.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Fish Audio")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "fishaudio"
    assert result.model == "s1"
    assert result.voice == _VOICE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_fishaudio_tts_url_and_headers(fishaudio_settings: Settings) -> None:
    ws = FakeWebSocket(_finish_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return ws

    provider = FishAudioTTSProvider(fishaudio_settings, model="s2.1-pro", voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.fishaudio.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert captured["url"] == "wss://api.fish.audio/v1/tts/live"
    headers = captured["kwargs"]["additional_headers"]  # type: ignore[index]
    assert headers["Authorization"] == "Bearer test-fishaudio-key"
    assert headers["model"] == "s2.1-pro"
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_fishaudio_tts_sends_start_text_stop(fishaudio_settings: Settings) -> None:
    ws = FakeWebSocket(_finish_events([make_pcm_bytes(240)]))
    provider = FishAudioTTSProvider(fishaudio_settings, model="s1", voice=_VOICE)

    with patch("coval_bench.providers.tts.fishaudio.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    sent = [ormsgpack.unpackb(m) for m in ws.sent if isinstance(m, bytes)]
    assert sent[0]["event"] == "start"
    request = sent[0]["request"]
    assert request["format"] == "pcm"
    assert request["sample_rate"] == 44100
    assert request["reference_id"] == _VOICE
    assert request["latency"] == "balanced"
    assert sent[1] == {"event": "text", "text": "Hello world"}
    assert sent[2] == {"event": "stop"}
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_fishaudio_tts_finish_error(fishaudio_settings: Settings) -> None:
    ws = FakeWebSocket([ormsgpack.packb({"event": "finish", "reason": "error"})])
    provider = FishAudioTTSProvider(fishaudio_settings, model="s1", voice=_VOICE)

    with patch("coval_bench.providers.tts.fishaudio.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_fishaudio_tts_skips_non_audio_events(fishaudio_settings: Settings) -> None:
    events: list[bytes | str] = [ormsgpack.packb({"event": "log", "message": "queued"})]
    events.extend(_finish_events([make_pcm_bytes(240)]))
    ws = FakeWebSocket(events)
    provider = FishAudioTTSProvider(fishaudio_settings, model="s1", voice=_VOICE)

    with patch("coval_bench.providers.tts.fishaudio.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_fishaudio_tts_ttfa_on_first_chunk(fishaudio_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_finish_events(chunks))
    provider = FishAudioTTSProvider(fishaudio_settings, model="s1", voice=_VOICE)

    times = iter([0.0, 0.1])

    with (
        patch(
            "coval_bench.providers.tts.fishaudio.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.fishaudio.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_fishaudio_tts_invalid_model_raises(fishaudio_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Fish Audio TTS model"):
        FishAudioTTSProvider(fishaudio_settings, model="not-a-model", voice=_VOICE)


def test_fishaudio_tts_missing_voice_raises(fishaudio_settings: Settings) -> None:
    with pytest.raises(ValueError, match="requires a voice"):
        FishAudioTTSProvider(fishaudio_settings, model="s1", voice="")


def test_fishaudio_tts_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="fishaudio_api_key is required"):
        FishAudioTTSProvider(_settings(fishaudio_api_key=None), model="s1", voice=_VOICE)


def test_fishaudio_tts_provider_name(fishaudio_settings: Settings) -> None:
    provider = FishAudioTTSProvider(fishaudio_settings, model="s1", voice=_VOICE)
    assert provider.name == "fishaudio-s1"
    assert provider.model == "s1"
