# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Soniox WebSocket TTS provider."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.soniox import _WS_URL, SonioxTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes


def _audio_events(pcm_chunks: list[bytes]) -> list[str]:
    events: list[str] = []
    for chunk in pcm_chunks:
        events.append(json.dumps({"audio": base64.b64encode(chunk).decode(), "stream_id": "probe"}))
    events.append(json.dumps({"terminated": True, "stream_id": "probe"}))
    return events


@pytest.mark.asyncio
async def test_soniox_tts_happy_path(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240)]))
    provider = SonioxTTSProvider(fake_settings, model="tts-rt-v1", voice="Adrian")

    with patch(
        "coval_bench.providers.tts.soniox.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("Hello from Soniox")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "soniox"
    assert result.model == "tts-rt-v1"
    assert result.voice == "Adrian"
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_soniox_tts_url_and_in_band_auth(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return ws

    provider = SonioxTTSProvider(fake_settings, model="tts-rt-v1", voice="Adrian")

    with patch(
        "coval_bench.providers.tts.soniox.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert captured["url"] == _WS_URL
    # Soniox authenticates in-band, not via an Authorization header.
    assert "additional_headers" not in captured["kwargs"]  # type: ignore[operator]

    config = json.loads(ws.sent[0])
    assert config["api_key"] == "test-soniox-key"
    assert config["model"] == "tts-rt-v1"
    assert config["voice"] == "Adrian"
    assert config["audio_format"] == "pcm_s16le"
    assert config["sample_rate"] == 24000


@pytest.mark.asyncio
async def test_soniox_tts_sends_config_then_text(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240)]))
    provider = SonioxTTSProvider(fake_settings, model="tts-rt-v1", voice="Adrian")

    with patch(
        "coval_bench.providers.tts.soniox.ws_client.connect",
        return_value=ws,
    ):
        await provider.synthesize("Hello world")

    sent_json = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert "api_key" in sent_json[0]
    assert sent_json[1]["text"] == "Hello world"
    assert sent_json[1]["text_end"] is True
    # All frames in a synthesis share one stream_id.
    assert sent_json[0]["stream_id"] == sent_json[1]["stream_id"]


@pytest.mark.asyncio
async def test_soniox_tts_error_event(fake_settings: Settings) -> None:
    ws = FakeWebSocket(
        [
            json.dumps(
                {
                    "stream_id": "probe",
                    "error_code": 400,
                    "error_type": "invalid_request",
                    "error_message": "invalid api key",
                }
            )
        ]
    )
    provider = SonioxTTSProvider(fake_settings, model="tts-rt-v1", voice="Adrian")

    with patch(
        "coval_bench.providers.tts.soniox.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "invalid api key" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_soniox_tts_error_after_partial_audio(fake_settings: Settings) -> None:
    chunk = make_pcm_bytes(240)
    events = [
        json.dumps({"audio": base64.b64encode(chunk).decode(), "stream_id": "probe"}),
        json.dumps({"error_code": 500, "error_message": "stream interrupted"}),
    ]
    ws = FakeWebSocket(events)
    provider = SonioxTTSProvider(fake_settings, model="tts-rt-v1", voice="Adrian")

    with patch(
        "coval_bench.providers.tts.soniox.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "stream interrupted" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is not None


@pytest.mark.asyncio
async def test_soniox_tts_ttfa_set_on_first_chunk_only(fake_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_audio_events(chunks))
    provider = SonioxTTSProvider(fake_settings, model="tts-rt-v1", voice="Adrian")

    times = iter([0.0, 0.1, 1.0, 2.0])

    with (
        patch(
            "coval_bench.providers.tts.soniox.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch(
            "coval_bench.providers.tts.soniox.ws_client.connect",
            return_value=ws,
        ),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_soniox_tts_skips_empty_audio(fake_settings: Settings) -> None:
    events = [
        json.dumps({"audio": "", "stream_id": "probe"}),
        json.dumps({"terminated": True, "stream_id": "probe"}),
    ]
    ws = FakeWebSocket(events)
    provider = SonioxTTSProvider(fake_settings, model="tts-rt-v1", voice="Adrian")

    with patch(
        "coval_bench.providers.tts.soniox.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


def test_soniox_tts_invalid_model_raises(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Soniox TTS model"):
        SonioxTTSProvider(fake_settings, model="not-a-model", voice="Adrian")


def test_soniox_tts_invalid_voice_raises(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Soniox TTS voice"):
        SonioxTTSProvider(fake_settings, model="tts-rt-v1", voice="not-a-voice")


def test_soniox_tts_missing_api_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        soniox_api_key=None,
    )
    with pytest.raises(ValueError, match="soniox_api_key is required"):
        SonioxTTSProvider(settings, model="tts-rt-v1", voice="Adrian")


def test_soniox_tts_provider_name(fake_settings: Settings) -> None:
    provider = SonioxTTSProvider(fake_settings, model="tts-rt-v1", voice="Adrian")
    assert provider.name == "soniox-tts-rt-v1"
    assert provider.model == "tts-rt-v1"
