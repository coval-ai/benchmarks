# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the xAI Grok WebSocket TTS provider."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.xai import XaiTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes


def _audio_events(pcm_chunks: list[bytes]) -> list[str]:
    events: list[str] = []
    for chunk in pcm_chunks:
        events.append(
            json.dumps({"type": "audio.delta", "delta": base64.b64encode(chunk).decode()})
        )
    events.append(json.dumps({"type": "audio.done", "trace_id": "test-trace-id"}))
    return events


@pytest.mark.asyncio
async def test_xai_tts_happy_path(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240)]))
    provider = XaiTTSProvider(fake_settings, model="grok-tts", voice="eve")

    with patch(
        "coval_bench.providers.tts.xai.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("Hello from Grok")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "xai"
    assert result.model == "grok-tts"
    assert result.voice == "eve"
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_xai_tts_url_and_auth(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(
        url: str,
        additional_headers: dict[str, str] | None = None,
        **_: object,
    ) -> FakeWebSocket:
        captured["url"] = url
        captured["headers"] = dict(additional_headers or {})
        return ws

    provider = XaiTTSProvider(fake_settings, model="grok-tts", voice="eve")

    with patch(
        "coval_bench.providers.tts.xai.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    parsed = urlparse(str(captured["url"]))
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.x.ai"
    assert parsed.path == "/v1/tts"
    query = parse_qs(parsed.query)
    assert query.get("language") == ["en"]
    assert query.get("voice") == ["eve"]
    assert query.get("codec") == ["pcm"]
    assert query.get("sample_rate") == ["24000"]
    assert query.get("text_normalization") == ["false"]
    assert query.get("optimize_streaming_latency") == ["2"]
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers.get("Authorization") == "Bearer test-xai-key"


@pytest.mark.asyncio
async def test_xai_tts_sends_text_delta_then_done(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240)]))
    provider = XaiTTSProvider(fake_settings, model="grok-tts", voice="eve")

    with patch(
        "coval_bench.providers.tts.xai.ws_client.connect",
        return_value=ws,
    ):
        await provider.synthesize("Hello world")

    sent_json = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert sent_json[0] == {"type": "text.delta", "delta": "Hello world"}
    assert sent_json[1] == {"type": "text.done"}


@pytest.mark.asyncio
async def test_xai_tts_error_event(fake_settings: Settings) -> None:
    ws = FakeWebSocket([json.dumps({"type": "error", "message": "invalid api key"})])
    provider = XaiTTSProvider(fake_settings, model="grok-tts", voice="eve")

    with patch(
        "coval_bench.providers.tts.xai.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "invalid api key" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_xai_tts_error_after_partial_audio(fake_settings: Settings) -> None:
    chunk = make_pcm_bytes(240)
    events = [
        json.dumps({"type": "audio.delta", "delta": base64.b64encode(chunk).decode()}),
        json.dumps({"type": "error", "message": "stream interrupted"}),
    ]
    ws = FakeWebSocket(events)
    provider = XaiTTSProvider(fake_settings, model="grok-tts", voice="eve")

    with patch(
        "coval_bench.providers.tts.xai.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "stream interrupted" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is not None


@pytest.mark.asyncio
async def test_xai_tts_ttfa_set_on_first_chunk_only(fake_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_audio_events(chunks))
    provider = XaiTTSProvider(fake_settings, model="grok-tts", voice="eve")

    times = iter([0.0, 0.1, 1.0, 2.0])

    with (
        patch(
            "coval_bench.providers.tts.xai.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch(
            "coval_bench.providers.tts.xai.ws_client.connect",
            return_value=ws,
        ),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_xai_tts_skips_empty_audio_delta(fake_settings: Settings) -> None:
    events = [
        json.dumps({"type": "audio.delta", "delta": ""}),
        json.dumps({"type": "audio.done", "trace_id": "test"}),
    ]
    ws = FakeWebSocket(events)
    provider = XaiTTSProvider(fake_settings, model="grok-tts", voice="eve")

    with patch(
        "coval_bench.providers.tts.xai.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


def test_xai_tts_invalid_model_raises(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid xAI TTS model"):
        XaiTTSProvider(fake_settings, model="not-a-model", voice="eve")


def test_xai_tts_invalid_voice_raises(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid xAI TTS voice"):
        XaiTTSProvider(fake_settings, model="grok-tts", voice="not-a-voice")


def test_xai_tts_missing_api_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        xai_api_key=None,
    )
    with pytest.raises(ValueError, match="xai_api_key is required"):
        XaiTTSProvider(settings, model="grok-tts", voice="eve")


def test_xai_tts_provider_name(fake_settings: Settings) -> None:
    provider = XaiTTSProvider(fake_settings, model="grok-tts", voice="eve")
    assert provider.name == "xai-grok-tts"
    assert provider.model == "grok-tts"
