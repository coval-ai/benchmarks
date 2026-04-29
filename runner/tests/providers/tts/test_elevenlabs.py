# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ElevenLabs TTS provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.elevenlabs import ElevenLabsTTSProvider

from .conftest import make_pcm_bytes

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    """ElevenLabs synthesize → ttfa set, WAV file written."""
    pcm_chunks = [make_pcm_bytes(240), make_pcm_bytes(240)]

    provider = ElevenLabsTTSProvider(
        fake_settings,
        model="eleven_flash_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
    )

    mock_sdk = MagicMock()
    mock_sdk.text_to_speech.convert.return_value = iter(pcm_chunks)

    with patch("coval_bench.providers.tts.elevenlabs.ElevenLabs", return_value=mock_sdk):
        result = await provider.synthesize("Hello from ElevenLabs")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "elevenlabs"
    assert result.model == "eleven_flash_v2_5"

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_elevenlabs_all_models(fake_settings: Settings) -> None:
    """Supported models all work with monkeypatched SDK."""
    from coval_bench.providers.tts.elevenlabs import SUPPORTED_MODELS

    pcm = make_pcm_bytes()
    for model in SUPPORTED_MODELS:
        provider = ElevenLabsTTSProvider(fake_settings, model=model, voice="test-voice")
        mock_sdk = MagicMock()
        mock_sdk.text_to_speech.convert.return_value = iter([pcm])
        with patch("coval_bench.providers.tts.elevenlabs.ElevenLabs", return_value=mock_sdk):
            result = await provider.synthesize("test")
        assert result.error is None, f"Model {model}: {result.error}"
        if result.audio_path:
            result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_sdk_error(fake_settings: Settings) -> None:
    """SDK exception → result.error populated, audio_path None."""
    provider = ElevenLabsTTSProvider(fake_settings, model="eleven_flash_v2_5", voice="test-voice")

    with patch(
        "coval_bench.providers.tts.elevenlabs.ElevenLabs",
        side_effect=RuntimeError("API rate limit"),
    ):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "API rate limit" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_elevenlabs_empty_response(fake_settings: Settings) -> None:
    """Empty iterator from SDK → audio_path is None, error is None."""
    provider = ElevenLabsTTSProvider(fake_settings, model="eleven_flash_v2_5", voice="test-voice")

    mock_sdk = MagicMock()
    mock_sdk.text_to_speech.convert.return_value = iter([])  # nothing

    with patch("coval_bench.providers.tts.elevenlabs.ElevenLabs", return_value=mock_sdk):
        result = await provider.synthesize("silence")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# Static method coverage — JSON-encoded audio branch
# ---------------------------------------------------------------------------


def test_elevenlabs_is_audio_chunk_bytes() -> None:
    """Bytes with content are audio chunks."""
    assert ElevenLabsTTSProvider._is_audio_chunk(b"\x00\x01") is True
    assert ElevenLabsTTSProvider._is_audio_chunk(b"") is False


def test_elevenlabs_is_audio_chunk_json_with_audio() -> None:
    """JSON-encoded message with 'audio' key is an audio chunk."""
    import base64
    import json

    payload = json.dumps({"audio": base64.b64encode(b"\x00\x01").decode()})
    assert ElevenLabsTTSProvider._is_audio_chunk(payload) is True


def test_elevenlabs_is_audio_chunk_json_no_audio() -> None:
    """JSON without 'audio' key is not an audio chunk."""
    import json

    payload = json.dumps({"isFinal": True})
    assert ElevenLabsTTSProvider._is_audio_chunk(payload) is False


def test_elevenlabs_is_audio_chunk_invalid_json() -> None:
    """Non-JSON string is not an audio chunk."""
    assert ElevenLabsTTSProvider._is_audio_chunk("not json") is False


def test_elevenlabs_extract_audio_from_json() -> None:
    """Extracts base64-encoded audio from JSON message."""
    import base64
    import json

    raw = b"\x00\x01\x02"
    payload = json.dumps({"audio": base64.b64encode(raw).decode()})
    result = ElevenLabsTTSProvider._extract_audio(payload)
    assert result == raw


def test_elevenlabs_extract_audio_fallback() -> None:
    """Invalid JSON returns empty bytes."""
    result = ElevenLabsTTSProvider._extract_audio("invalid json {{{")
    assert result == b""


@pytest.mark.asyncio
async def test_elevenlabs_json_audio_chunks(fake_settings: Settings) -> None:
    """Synthesize with JSON-encoded audio chunk (ElevenLabs WS format)."""
    import base64
    import json

    raw_pcm = make_pcm_bytes(240)
    json_chunk = json.dumps({"audio": base64.b64encode(raw_pcm).decode()})

    provider = ElevenLabsTTSProvider(fake_settings, model="eleven_flash_v2_5", voice="test-voice")

    mock_sdk = MagicMock()
    mock_sdk.text_to_speech.convert.return_value = iter([json_chunk])

    with patch("coval_bench.providers.tts.elevenlabs.ElevenLabs", return_value=mock_sdk):
        result = await provider.synthesize("json audio test")

    # The json branch goes through _extract_audio which returns the decoded bytes
    # audio stream write receives empty string from non-bytes path, ttfa may be None
    assert result.error is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_elevenlabs_name_and_model(fake_settings: Settings) -> None:
    p = ElevenLabsTTSProvider(fake_settings, model="eleven_turbo_v2_5", voice="voice-id")
    assert p.name == "elevenlabs-eleven_turbo_v2_5"
    assert p.model == "eleven_turbo_v2_5"


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_elevenlabs_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",  # type: ignore[arg-type]
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        elevenlabs_api_key=None,
    )
    with pytest.raises(ValueError, match="elevenlabs_api_key"):
        ElevenLabsTTSProvider(settings_no_key, model="eleven_flash_v2_5", voice="v")
