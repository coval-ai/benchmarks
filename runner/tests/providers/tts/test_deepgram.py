# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Deepgram TTS provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.deepgram import DeepgramTTSProvider

from .conftest import FakeAiohttpResponse, FakeAiohttpSession, make_pcm_bytes

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    """Deepgram HTTP streaming → ttfa set, WAV written."""
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240)]
    provider = DeepgramTTSProvider(
        fake_settings, model="aura-2-thalia-en", voice="aura-2-thalia-en"
    )

    fake_resp = FakeAiohttpResponse(chunks, status=200)
    fake_sess = FakeAiohttpSession(fake_resp)

    with patch("coval_bench.providers.tts.deepgram.aiohttp.ClientSession", return_value=fake_sess):
        result = await provider.synthesize("Hello from Deepgram")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "deepgram"
    assert result.model == "aura-2-thalia-en"

    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Error path — HTTP error status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_http_error_status(fake_settings: Settings) -> None:
    """Non-200 response → result.error populated, audio_path None."""
    provider = DeepgramTTSProvider(
        fake_settings, model="aura-2-thalia-en", voice="aura-2-thalia-en"
    )

    fake_resp = FakeAiohttpResponse([], status=403, text_body="Forbidden")
    fake_sess = FakeAiohttpSession(fake_resp)

    with patch("coval_bench.providers.tts.deepgram.aiohttp.ClientSession", return_value=fake_sess):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "403" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_deepgram_network_error(fake_settings: Settings) -> None:
    """Network exception → result.error populated, audio_path None."""
    provider = DeepgramTTSProvider(
        fake_settings, model="aura-2-thalia-en", voice="aura-2-thalia-en"
    )

    with patch(
        "coval_bench.providers.tts.deepgram.aiohttp.ClientSession",
        side_effect=RuntimeError("network error"),
    ):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert result.audio_path is None


# ---------------------------------------------------------------------------
# Empty audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_no_audio_chunks(fake_settings: Settings) -> None:
    """Empty response body → audio_path None, no error."""
    provider = DeepgramTTSProvider(
        fake_settings, model="aura-2-thalia-en", voice="aura-2-thalia-en"
    )

    fake_resp = FakeAiohttpResponse([], status=200)
    fake_sess = FakeAiohttpSession(fake_resp)

    with patch("coval_bench.providers.tts.deepgram.aiohttp.ClientSession", return_value=fake_sess):
        result = await provider.synthesize("silent test")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_deepgram_name_and_model(fake_settings: Settings) -> None:
    p = DeepgramTTSProvider(fake_settings, model="aura-2-thalia-en", voice="v")
    assert p.name == "deepgram-aura-2-thalia-en"
    assert p.model == "aura-2-thalia-en"


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Re-activated 2026-04-30: aura-2-thalia-en (Deepgram production voice).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_aura_2_thalia_en_http(fake_settings: Settings) -> None:
    """aura-2-thalia-en HTTP streaming → ttfa set, valid WAV with .wav magic bytes."""
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    provider = DeepgramTTSProvider(
        fake_settings,
        model="aura-2-thalia-en",
        voice="aura-2-thalia-en",
    )

    fake_resp = FakeAiohttpResponse(chunks, status=200)
    fake_sess = FakeAiohttpSession(fake_resp)
    with patch("coval_bench.providers.tts.deepgram.aiohttp.ClientSession", return_value=fake_sess):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.provider == "deepgram"
    assert result.model == "aura-2-thalia-en"
    assert result.voice == "aura-2-thalia-en"
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    result.audio_path.unlink()


def test_deepgram_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",  # type: ignore[arg-type]
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        deepgram_api_key=None,
    )
    with pytest.raises(ValueError, match="deepgram_api_key"):
        DeepgramTTSProvider(settings_no_key, model="aura-2-thalia-en", voice="v")
