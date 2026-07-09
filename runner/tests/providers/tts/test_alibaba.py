# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Alibaba Cloud DashScope realtime TTS provider (Qwen3-TTS-Flash)."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts.alibaba import AlibabaTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_MODEL = "qwen3-tts-flash-realtime"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql://runner:password@localhost:5432/benchmarks",
        "dataset_bucket": "test-bucket",
        "dataset_id": "stt-v1",
        "runner_sha": "test",
        "log_level": "DEBUG",
        "alibaba_api_key": SecretStr("test-alibaba-key"),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _delta_event(pcm: bytes) -> str:
    return json.dumps(
        {"type": "response.audio.delta", "delta": base64.b64encode(pcm).decode("ascii")}
    )


def _session_events(pcm_chunks: list[bytes]) -> list[str | bytes]:
    events: list[str | bytes] = [
        json.dumps({"type": "session.created", "session": {"id": "sess-1"}})
    ]
    events.extend(_delta_event(pcm) for pcm in pcm_chunks)
    events.append(json.dumps({"type": "response.done"}))
    events.append(json.dumps({"type": "session.finished"}))
    return events


@pytest.fixture()
def alibaba_settings() -> Settings:
    return _settings()


@pytest.mark.asyncio
async def test_alibaba_tts_happy_path(alibaba_settings: Settings) -> None:
    ws = FakeWebSocket(_session_events([make_pcm_bytes(240)]))
    provider = AlibabaTTSProvider(alibaba_settings, model=_MODEL, voice="Cherry")

    with patch("coval_bench.providers.tts.alibaba.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Alibaba")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "alibaba"
    assert result.model == _MODEL
    assert result.voice == "Cherry"
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_alibaba_tts_url_and_header_auth(alibaba_settings: Settings) -> None:
    ws = FakeWebSocket(_session_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return ws

    provider = AlibabaTTSProvider(alibaba_settings, model=_MODEL, voice="Cherry")

    with patch(
        "coval_bench.providers.tts.alibaba.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert captured["url"] == (
        f"wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model={_MODEL}"
    )
    headers = captured["kwargs"]["additional_headers"]  # type: ignore[index]
    assert headers["Authorization"] == "Bearer test-alibaba-key"
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_alibaba_tts_url_override(alibaba_settings: Settings) -> None:
    ws = FakeWebSocket(_session_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        return ws

    override = "wss://ws-1234.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime"
    provider = AlibabaTTSProvider(_settings(alibaba_tts_url=override), model=_MODEL, voice="Cherry")

    with patch(
        "coval_bench.providers.tts.alibaba.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert captured["url"] == f"{override}?model={_MODEL}"
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_alibaba_tts_url_override_with_query_string(alibaba_settings: Settings) -> None:
    ws = FakeWebSocket(_session_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        return ws

    override = "wss://gateway.example.com/api-ws/v1/realtime?token=abc"
    provider = AlibabaTTSProvider(_settings(alibaba_tts_url=override), model=_MODEL, voice="Cherry")

    with patch(
        "coval_bench.providers.tts.alibaba.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert captured["url"] == f"{override}&model={_MODEL}"
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_alibaba_tts_sends_update_append_commit_finish(
    alibaba_settings: Settings,
) -> None:
    ws = FakeWebSocket(_session_events([make_pcm_bytes(240)]))
    provider = AlibabaTTSProvider(alibaba_settings, model=_MODEL, voice="Ethan")

    with patch("coval_bench.providers.tts.alibaba.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent[0]["type"] == "session.update"
    assert sent[0]["session"]["voice"] == "Ethan"
    assert sent[0]["session"]["mode"] == "commit"
    assert sent[0]["session"]["response_format"] == "pcm"
    assert sent[0]["session"]["sample_rate"] == 24000
    assert sent[1] == {"type": "input_text_buffer.append", "text": "Hello world"}
    assert sent[2] == {"type": "input_text_buffer.commit"}
    assert sent[3] == {"type": "session.finish"}
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_alibaba_tts_error_event(alibaba_settings: Settings) -> None:
    ws = FakeWebSocket([json.dumps({"type": "error", "error": {"message": "bad request"}})])
    provider = AlibabaTTSProvider(alibaba_settings, model=_MODEL, voice="Cherry")

    with patch("coval_bench.providers.tts.alibaba.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error == "bad request"
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_alibaba_tts_ttfa_on_first_delta(alibaba_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_session_events(chunks))
    provider = AlibabaTTSProvider(alibaba_settings, model=_MODEL, voice="Cherry")

    times = iter([0.0, 0.1, 1.0, 2.0])

    with (
        patch(
            "coval_bench.providers.tts.alibaba.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.alibaba.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_alibaba_tts_invalid_model_raises(alibaba_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Alibaba TTS model"):
        AlibabaTTSProvider(alibaba_settings, model="not-a-model", voice="Cherry")


def test_alibaba_tts_invalid_voice_raises(alibaba_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Alibaba TTS voice"):
        AlibabaTTSProvider(alibaba_settings, model=_MODEL, voice="not-a-voice")


def test_alibaba_tts_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="alibaba_api_key is required"):
        AlibabaTTSProvider(_settings(alibaba_api_key=None), model=_MODEL, voice="Cherry")


def test_alibaba_tts_provider_name(alibaba_settings: Settings) -> None:
    provider = AlibabaTTSProvider(alibaba_settings, model=_MODEL, voice="Cherry")
    assert provider.name == "alibaba"
    assert provider.model == _MODEL
