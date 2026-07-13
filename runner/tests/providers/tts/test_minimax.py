# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MiniMax WebSocket TTS provider."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts.minimax import MinimaxTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_VOICE = "English_expressive_narrator"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql://runner:password@localhost:5432/benchmarks",
        "dataset_bucket": "test-bucket",
        "dataset_id": "stt-v1",
        "runner_sha": "test",
        "log_level": "DEBUG",
        "minimax_api_key": SecretStr("test-minimax-key"),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _ok(event: str, **extra: object) -> str:
    return json.dumps({"event": event, "base_resp": {"status_code": 0}, **extra})


def _session_events(pcm_chunks: list[bytes]) -> list[bytes | str]:
    events: list[bytes | str] = [_ok("connected_success"), _ok("task_started")]
    events.extend(_ok("task_continued", data={"audio": chunk.hex()}) for chunk in pcm_chunks)
    events.append(_ok("task_finished"))
    return events


@pytest.fixture()
def minimax_settings() -> Settings:
    return _settings()


@pytest.mark.asyncio
async def test_minimax_tts_happy_path(minimax_settings: Settings) -> None:
    ws = FakeWebSocket(_session_events([make_pcm_bytes(240)]))
    provider = MinimaxTTSProvider(minimax_settings, model="speech-2.8-hd", voice=_VOICE)

    with patch("coval_bench.providers.tts.minimax.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from MiniMax")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "minimax"
    assert result.model == "speech-2.8-hd"
    assert result.voice == _VOICE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_minimax_tts_url_and_headers(minimax_settings: Settings) -> None:
    ws = FakeWebSocket(_session_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return ws

    provider = MinimaxTTSProvider(minimax_settings, model="speech-2.8-turbo", voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.minimax.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert captured["url"] == "wss://api.minimax.io/ws/v1/t2a_v2"
    headers = captured["kwargs"]["additional_headers"]  # type: ignore[index]
    assert headers["Authorization"] == "Bearer test-minimax-key"
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_minimax_tts_sends_start_continue_finish(minimax_settings: Settings) -> None:
    ws = FakeWebSocket(_session_events([make_pcm_bytes(240)]))
    provider = MinimaxTTSProvider(minimax_settings, model="speech-2.8-hd", voice=_VOICE)

    with patch("coval_bench.providers.tts.minimax.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent[0]["event"] == "task_start"
    assert sent[0]["model"] == "speech-2.8-hd"
    assert sent[0]["voice_setting"] == {"voice_id": _VOICE}
    assert sent[0]["audio_setting"] == {"format": "pcm", "sample_rate": 44100, "channel": 1}
    assert sent[1] == {"event": "task_continue", "text": "Hello world"}
    assert sent[2] == {"event": "task_finish"}
    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_minimax_tts_task_failed(minimax_settings: Settings) -> None:
    events: list[bytes | str] = [
        _ok("connected_success"),
        json.dumps(
            {
                "event": "task_failed",
                "base_resp": {"status_code": 1004, "status_msg": "authentication failed"},
            }
        ),
    ]
    ws = FakeWebSocket(events)
    provider = MinimaxTTSProvider(minimax_settings, model="speech-2.8-hd", voice=_VOICE)

    with patch("coval_bench.providers.tts.minimax.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "1004" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_minimax_tts_nonzero_status_code(minimax_settings: Settings) -> None:
    events: list[bytes | str] = [
        _ok("connected_success"),
        _ok("task_started"),
        json.dumps(
            {
                "event": "task_continued",
                "base_resp": {"status_code": 1002, "status_msg": "rate limit"},
            }
        ),
    ]
    ws = FakeWebSocket(events)
    provider = MinimaxTTSProvider(minimax_settings, model="speech-2.8-hd", voice=_VOICE)

    with patch("coval_bench.providers.tts.minimax.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "1002" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_minimax_tts_skips_empty_audio_payloads(minimax_settings: Settings) -> None:
    events: list[bytes | str] = [
        _ok("connected_success"),
        _ok("task_started"),
        _ok("task_continued"),
        _ok("task_continued", data={}),
        _ok("task_continued", data={"audio": ""}),
        _ok("task_continued", data={"audio": make_pcm_bytes(240).hex()}),
        _ok("task_finished"),
    ]
    ws = FakeWebSocket(events)
    provider = MinimaxTTSProvider(minimax_settings, model="speech-2.8-hd", voice=_VOICE)

    with patch("coval_bench.providers.tts.minimax.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_minimax_tts_skips_binary_frames(minimax_settings: Settings) -> None:
    events: list[bytes | str] = [
        _ok("connected_success"),
        _ok("task_started"),
        b"\x00\x01\x02",
        _ok("task_continued", data={"audio": make_pcm_bytes(240).hex()}),
        _ok("task_finished"),
    ]
    ws = FakeWebSocket(events)
    provider = MinimaxTTSProvider(minimax_settings, model="speech-2.8-hd", voice=_VOICE)

    with patch("coval_bench.providers.tts.minimax.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_minimax_tts_stops_on_is_final(minimax_settings: Settings) -> None:
    events: list[bytes | str] = [
        _ok("connected_success"),
        _ok("task_started"),
        _ok("task_continued", data={"audio": make_pcm_bytes(240).hex()}, is_final=True),
    ]
    ws = FakeWebSocket(events)
    provider = MinimaxTTSProvider(minimax_settings, model="speech-2.8-hd", voice=_VOICE)

    with patch("coval_bench.providers.tts.minimax.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_minimax_tts_ttfa_on_first_chunk(minimax_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_session_events(chunks))
    provider = MinimaxTTSProvider(minimax_settings, model="speech-2.8-hd", voice=_VOICE)

    times = iter([0.0, 0.1])

    with (
        patch(
            "coval_bench.providers.tts.minimax.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.minimax.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_minimax_tts_invalid_model_raises(minimax_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid MiniMax TTS model"):
        MinimaxTTSProvider(minimax_settings, model="not-a-model", voice=_VOICE)


def test_minimax_tts_missing_voice_raises(minimax_settings: Settings) -> None:
    with pytest.raises(ValueError, match="requires a voice"):
        MinimaxTTSProvider(minimax_settings, model="speech-2.8-hd", voice="")


def test_minimax_tts_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="minimax_api_key is required"):
        MinimaxTTSProvider(_settings(minimax_api_key=None), model="speech-2.8-hd", voice=_VOICE)


def test_minimax_tts_provider_name(minimax_settings: Settings) -> None:
    provider = MinimaxTTSProvider(minimax_settings, model="speech-2.8-hd", voice=_VOICE)
    assert provider.name == "minimax-speech-2.8-hd"
    assert provider.model == "speech-2.8-hd"
