# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Gradium WebSocket TTS provider."""

from __future__ import annotations

import base64
import json
import wave
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.gradium import GradiumTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes


def _audio_events(pcm_chunks: list[bytes]) -> list[str]:
    """Build a Gradium event stream: ready, audio messages, then end_of_stream."""
    events: list[str] = [json.dumps({"type": "ready"})]
    for chunk in pcm_chunks:
        events.append(json.dumps({"type": "audio", "audio": base64.b64encode(chunk).decode()}))
    events.append(json.dumps({"type": "end_of_stream"}))
    return events


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_happy_path(fake_settings: Settings) -> None:
    """Synthesize sets TTFA and writes a valid WAV file."""
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240)]))
    provider = GradiumTTSProvider(fake_settings, model="default", voice="YTpq7expH9539ERJ")

    with patch(
        "coval_bench.providers.tts.gradium.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("Hello from Gradium")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "gradium"
    assert result.model == "default"
    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Sample rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_sample_rate(fake_settings: Settings) -> None:
    """WAV output is written at 48 kHz, matching the Gradium PCM spec."""
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240)]))
    provider = GradiumTTSProvider(fake_settings, model="default", voice="YTpq7expH9539ERJ")

    with patch(
        "coval_bench.providers.tts.gradium.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("sample rate test")

    assert result.audio_path is not None
    with wave.open(str(result.audio_path), "rb") as wav_file:
        assert wav_file.getframerate() == 48000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Protocol: sent messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_sends_setup_text_eos(fake_settings: Settings) -> None:
    """Provider sends setup, then text, then end_of_stream in that order."""
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240)]))
    provider = GradiumTTSProvider(fake_settings, model="default", voice="test-voice-id")

    with patch(
        "coval_bench.providers.tts.gradium.ws_client.connect",
        return_value=ws,
    ):
        await provider.synthesize("test text")

    assert len(ws.sent) >= 3
    setup = json.loads(ws.sent[0])
    text_msg = json.loads(ws.sent[1])
    eos_msg = json.loads(ws.sent[2])
    assert setup == {
        "type": "setup",
        "voice_id": "test-voice-id",
        "model_name": "default",
        "output_format": "pcm",
    }
    assert text_msg == {"type": "text", "text": "test text"}
    assert eos_msg == {"type": "end_of_stream"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_auth_header(fake_settings: Settings) -> None:
    """API key is passed as x-api-key header on connection."""
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240)]))
    captured: dict[str, str] = {}

    def _capture(
        url: str, additional_headers: dict[str, str] | None = None, **_kw: object
    ) -> FakeWebSocket:
        if additional_headers:
            captured.update(additional_headers)
        return ws

    provider = GradiumTTSProvider(fake_settings, model="default", voice="test-voice-id")

    with patch("coval_bench.providers.tts.gradium.ws_client.connect", side_effect=_capture):
        await provider.synthesize("auth test")

    assert captured.get("x-api-key") == "test-gradium-tts-key"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_server_error(fake_settings: Settings) -> None:
    """Server error frame populates result.error."""
    events = [
        json.dumps({"type": "ready"}),
        json.dumps({"type": "error", "message": "voice not found"}),
    ]
    ws = FakeWebSocket(events)
    provider = GradiumTTSProvider(fake_settings, model="default", voice="bad-voice")

    with patch(
        "coval_bench.providers.tts.gradium.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "voice not found" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_gradium_connect_failure(fake_settings: Settings) -> None:
    """WebSocket connect failure populates result.error."""
    provider = GradiumTTSProvider(fake_settings, model="default", voice="test-voice-id")

    with patch(
        "coval_bench.providers.tts.gradium.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "refused" in result.error.lower()
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_gradium_empty_response(fake_settings: Settings) -> None:
    """end_of_stream with no audio chunks leaves audio_path None."""
    events = [json.dumps({"type": "ready"}), json.dumps({"type": "end_of_stream"})]
    ws = FakeWebSocket(events)
    provider = GradiumTTSProvider(fake_settings, model="default", voice="test-voice-id")

    with patch(
        "coval_bench.providers.tts.gradium.ws_client.connect",
        return_value=ws,
    ):
        result = await provider.synthesize("silence")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# TTFA correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gradium_ttfa_first_chunk_only(fake_settings: Settings) -> None:
    """ttfa_ms is measured at the first audio chunk and not overwritten."""
    ws = FakeWebSocket(_audio_events([make_pcm_bytes(240), make_pcm_bytes(240)]))
    provider = GradiumTTSProvider(fake_settings, model="default", voice="test-voice-id")
    # asyncio.wait_for internally calls loop.time() twice; pad leading values.
    clock = iter([0.0, 0.0, 0.1, 0.2, 0.6])

    with (
        patch(
            "coval_bench.providers.tts.gradium.ws_client.connect",
            return_value=ws,
        ),
        patch("coval_bench.providers.tts.gradium.time.monotonic", side_effect=clock),
    ):
        result = await provider.synthesize("two chunks")

    assert result.error is None
    # start=0.1, first chunk=0.2 → 100ms; second chunk not measured (guard).
    assert result.ttfa_ms == pytest.approx(100.0)
    if result.audio_path:
        result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_gradium_name_and_model(fake_settings: Settings) -> None:
    p = GradiumTTSProvider(fake_settings, model="default", voice="v")
    assert p.name == "gradium"
    assert p.model == "default"


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_gradium_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        gradium_tts_api_key=None,
    )
    with pytest.raises(ValueError, match="gradium_tts_api_key"):
        GradiumTTSProvider(settings_no_key, model="default", voice="v")
