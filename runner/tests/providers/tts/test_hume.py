# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hume WebSocket TTS provider."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from coval_bench.config import Settings  # noqa: E402
from coval_bench.providers.tts.hume import SAMPLE_RATE, HumeTTSProvider  # noqa: E402

from .conftest import FakeWebSocket, make_pcm_bytes  # noqa: E402

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hume_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    """Hume octave-tts synthesize sets TTFA and writes WAV output."""
    pcm_chunks: list[str | bytes] = [make_pcm_bytes(240)]
    provider = HumeTTSProvider(
        fake_settings, model="octave-tts", voice="176a55b1-4468-4736-8878-db82729667c1"
    )
    ws = FakeWebSocket(pcm_chunks)

    with patch("coval_bench.providers.tts.hume.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Hume")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "hume"
    assert result.model == "octave-tts"

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_hume_octave_2_model(fake_settings: Settings) -> None:
    """octave-2 model also succeeds."""
    provider = HumeTTSProvider(fake_settings, model="octave-2", voice="test-voice-id")
    ws = FakeWebSocket([make_pcm_bytes(240)])

    with patch("coval_bench.providers.tts.hume.ws_client.connect", return_value=ws):
        result = await provider.synthesize("test octave-2")

    assert result.error is None
    if result.audio_path:
        result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Protocol: sent messages and URL shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hume_sends_text_flush_and_close(fake_settings: Settings) -> None:
    """Provider sends text, flush, then close messages in order."""
    ws = FakeWebSocket([make_pcm_bytes(240)])
    provider = HumeTTSProvider(fake_settings, model="octave-tts", voice="test-voice-id")

    with patch("coval_bench.providers.tts.hume.ws_client.connect", return_value=ws):
        await provider.synthesize("test text")

    assert len(ws.sent) >= 3
    first = json.loads(ws.sent[0])
    second = json.loads(ws.sent[1])
    third = json.loads(ws.sent[2])
    assert first == {
        "text": "test text",
        "voice": {"id": "test-voice-id", "provider": "HUME_AI"},
        "speed": 1.0,
        "trailing_silence": 0,
    }
    assert second == {"flush": True}
    assert third == {"close": True}


@pytest.mark.asyncio
async def test_hume_url_shape(fake_settings: Settings) -> None:
    """WS URL targets Hume input streaming endpoint with PCM params."""
    ws = FakeWebSocket([make_pcm_bytes(240)])
    captured: dict[str, str] = {}

    def _capture(url: str, **_kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        return ws

    provider = HumeTTSProvider(fake_settings, model="octave-2", voice="test-voice-id")

    with patch("coval_bench.providers.tts.hume.ws_client.connect", side_effect=_capture):
        await provider.synthesize("url test")

    parsed = urlparse(captured["url"])
    qs = parse_qs(parsed.query)

    assert parsed.hostname == "api.hume.ai"
    assert parsed.path == "/v0/tts/stream/input"
    assert qs["api_key"] == ["test-hume-key"]
    assert qs["instant_mode"] == ["true"]
    assert qs["format_type"] == ["pcm"]
    assert qs["strip_headers"] == ["true"]
    assert qs["version"] == ["2"]


@pytest.mark.asyncio
async def test_hume_voice_fallback(fake_settings: Settings) -> None:
    """voice=None falls back to the built-in Hume voice ID in the text message."""
    ws = FakeWebSocket([make_pcm_bytes(240)])
    provider = HumeTTSProvider(fake_settings, model="octave-tts", voice="")

    with patch("coval_bench.providers.tts.hume.ws_client.connect", return_value=ws):
        await provider.synthesize("voice fallback")

    first = json.loads(ws.sent[0])
    assert first["voice"] == {
        "id": "176a55b1-4468-4736-8878-db82729667c1",
        "provider": "HUME_AI",
    }


# ---------------------------------------------------------------------------
# Unsupported model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hume_unsupported_model(fake_settings: Settings) -> None:
    """Unsupported model returns error result without calling SDK."""
    provider = HumeTTSProvider(fake_settings, model="emphatic-voice-interface", voice="test-voice")
    result = await provider.synthesize("test")
    assert result.error is not None
    assert "Unsupported" in result.error
    assert result.audio_path is None


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hume_sdk_error(fake_settings: Settings) -> None:
    """WebSocket exception populates result.error and leaves audio_path empty."""
    provider = HumeTTSProvider(fake_settings, model="octave-tts", voice="test-voice-id")

    with patch(
        "coval_bench.providers.tts.hume.ws_client.connect",
        side_effect=RuntimeError("Hume API error"),
    ):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "Hume API error" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_hume_empty_response(fake_settings: Settings) -> None:
    """No audio chunks → audio_path is None."""
    provider = HumeTTSProvider(fake_settings, model="octave-tts", voice="test-voice-id")
    ws = FakeWebSocket([])

    with patch("coval_bench.providers.tts.hume.ws_client.connect", return_value=ws):
        result = await provider.synthesize("silence")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# TTFA correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hume_ttfa_set_on_first_chunk_only(fake_settings: Settings) -> None:
    """ttfa_ms is measured at the first audio chunk and not overwritten."""
    ws = FakeWebSocket([make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)])
    provider = HumeTTSProvider(fake_settings, model="octave-tts", voice="test-voice-id")
    # asyncio.timeout() may call monotonic before t0; pad leading values.
    clock = iter([0.0, 0.0, 0.0, 0.1, 0.5, 0.9])

    with (
        patch("coval_bench.providers.tts.hume.ws_client.connect", return_value=ws),
        patch("coval_bench.providers.tts.hume.time.monotonic", side_effect=clock),
    ):
        result = await provider.synthesize("three chunks")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    if result.audio_path:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_hume_empty_chunk_not_counted(fake_settings: Settings) -> None:
    """Empty binary messages must not set TTFA or write a WAV file."""
    ws = FakeWebSocket([b""])
    provider = HumeTTSProvider(fake_settings, model="octave-tts", voice="test-voice-id")

    with patch("coval_bench.providers.tts.hume.ws_client.connect", return_value=ws):
        result = await provider.synthesize("empty chunk")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_hume_session_timeout(fake_settings: Settings) -> None:
    """A stalled WebSocket session returns a clear timeout error."""
    import asyncio

    class StalledWebSocket(FakeWebSocket):
        def __aiter__(self) -> StalledWebSocket:
            return self

        async def __anext__(self) -> bytes:
            await asyncio.sleep(3600)
            return b""

    provider = HumeTTSProvider(fake_settings, model="octave-tts", voice="test-voice-id")
    ws = StalledWebSocket([])

    with (
        patch("coval_bench.providers.tts.hume.ws_client.connect", return_value=ws),
        patch("coval_bench.providers.tts.hume._WS_SESSION_TIMEOUT_S", 0.05),
    ):
        result = await provider.synthesize("stall")

    assert result.error is not None
    assert "timed out" in result.error.lower()
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_hume_server_error_event(fake_settings: Settings) -> None:
    """A server error frame populates result.error."""
    ws = FakeWebSocket([json.dumps({"type": "error", "message": "rate limit exceeded"})])
    provider = HumeTTSProvider(fake_settings, model="octave-tts", voice="test-voice-id")

    with patch("coval_bench.providers.tts.hume.ws_client.connect", return_value=ws):
        result = await provider.synthesize("error frame")

    assert result.error is not None
    assert "rate limit exceeded" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_hume_write_wav_sample_rate(fake_settings: Settings) -> None:
    """Hume writes raw PCM chunks as 48 kHz WAV output."""
    provider = HumeTTSProvider(fake_settings, model="octave-tts", voice="test-voice-id")
    ws = FakeWebSocket([make_pcm_bytes(240)])

    with patch("coval_bench.providers.tts.hume.ws_client.connect", return_value=ws):
        result = await provider.synthesize("sample rate")

    assert result.audio_path is not None
    with wave.open(str(result.audio_path), "rb") as wav_file:
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_hume_name_and_model(fake_settings: Settings) -> None:
    p = HumeTTSProvider(fake_settings, model="octave-tts", voice="v")
    assert p.name == "hume-octave-tts"
    assert p.model == "octave-tts"


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_hume_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        hume_api_key=None,
    )
    with pytest.raises(ValueError, match="hume_api_key"):
        HumeTTSProvider(settings_no_key, model="octave-tts", voice="v")
