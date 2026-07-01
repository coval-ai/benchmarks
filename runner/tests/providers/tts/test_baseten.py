# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Baseten WebSocket TTS provider (Qwen3-TTS)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts.baseten import BasetenTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_WS_URL = "wss://model-test.api.baseten.co/environments/production/websocket"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql://runner:password@localhost:5432/benchmarks",
        "dataset_bucket": "test-bucket",
        "dataset_id": "stt-v1",
        "runner_sha": "test",
        "log_level": "DEBUG",
        "baseten_api_key": SecretStr("test-baseten-key"),
        "baseten_qwen_url": _WS_URL,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _done_events(pcm_chunks: list[bytes]) -> list[bytes | str]:
    events: list[bytes | str] = list(pcm_chunks)
    events.append(json.dumps({"type": "session.done"}))
    return events


@pytest.fixture()
def baseten_settings() -> Settings:
    return _settings()


@pytest.mark.asyncio
async def test_baseten_tts_happy_path(baseten_settings: Settings) -> None:
    ws = FakeWebSocket(_done_events([make_pcm_bytes(240)]))
    provider = BasetenTTSProvider(baseten_settings, model="qwen3-tts-1.7b", voice="lisa")

    with patch("coval_bench.providers.tts.baseten.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Baseten")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "baseten"
    assert result.model == "qwen3-tts-1.7b"
    assert result.voice == "lisa"
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_baseten_tts_url_and_header_auth(baseten_settings: Settings) -> None:
    ws = FakeWebSocket(_done_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return ws

    provider = BasetenTTSProvider(baseten_settings, model="qwen3-tts-1.7b", voice="lisa")

    with patch(
        "coval_bench.providers.tts.baseten.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert captured["url"] == _WS_URL
    headers = captured["kwargs"]["additional_headers"]  # type: ignore[index]
    assert headers["Authorization"] == "Api-Key test-baseten-key"
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_baseten_tts_sends_config_text_done(baseten_settings: Settings) -> None:
    ws = FakeWebSocket(_done_events([make_pcm_bytes(240)]))
    provider = BasetenTTSProvider(baseten_settings, model="qwen3-tts-1.7b", voice="jim")

    with patch("coval_bench.providers.tts.baseten.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent[0]["type"] == "session.config"
    assert sent[0]["voice"] == "jim"
    assert sent[0]["response_format"] == "pcm"
    assert sent[0]["stream_audio"] is True
    assert sent[1] == {"type": "input.text", "text": "Hello world"}
    assert sent[2] == {"type": "input.done"}
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_baseten_tts_error_event(baseten_settings: Settings) -> None:
    ws = FakeWebSocket([json.dumps({"type": "error", "message": "bad request"})])
    provider = BasetenTTSProvider(baseten_settings, model="qwen3-tts-1.7b", voice="lisa")

    with patch("coval_bench.providers.tts.baseten.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "bad request" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_baseten_tts_ttfa_on_first_chunk(baseten_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_done_events(chunks))
    provider = BasetenTTSProvider(baseten_settings, model="qwen3-tts-1.7b", voice="lisa")

    times = iter([0.0, 0.1, 1.0, 2.0])

    with (
        patch(
            "coval_bench.providers.tts.baseten.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.baseten.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_baseten_tts_invalid_model_raises(baseten_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Baseten TTS model"):
        BasetenTTSProvider(baseten_settings, model="not-a-model", voice="lisa")


def test_baseten_tts_invalid_voice_raises(baseten_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Baseten TTS voice"):
        BasetenTTSProvider(baseten_settings, model="qwen3-tts-1.7b", voice="not-a-voice")


def test_baseten_tts_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="baseten_api_key is required"):
        BasetenTTSProvider(_settings(baseten_api_key=None), model="qwen3-tts-1.7b", voice="lisa")


def test_baseten_tts_missing_url_raises() -> None:
    with pytest.raises(ValueError, match="baseten_qwen_url is required"):
        BasetenTTSProvider(_settings(baseten_qwen_url=None), model="qwen3-tts-1.7b", voice="lisa")


def test_baseten_tts_provider_name(baseten_settings: Settings) -> None:
    provider = BasetenTTSProvider(baseten_settings, model="qwen3-tts-1.7b", voice="lisa")
    assert provider.name == "baseten-qwen3-tts-1.7b"
    assert provider.model == "qwen3-tts-1.7b"
