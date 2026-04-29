# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Cartesia TTS provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.cartesia import CartesiaTTSProvider

from .conftest import make_fake_cartesia_client, make_pcm_bytes

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cartesia_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    """Cartesia WS synthesize → ttfa set, valid WAV written."""
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240)]
    provider = CartesiaTTSProvider(
        fake_settings, model="sonic-3", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"
    )

    fake_client = make_fake_cartesia_client(chunks)

    with patch("coval_bench.providers.tts.cartesia.AsyncCartesia", return_value=fake_client):
        result = await provider.synthesize("Hello from Cartesia")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "cartesia"
    assert result.model == "sonic-3"

    # Provider must not auto-delete — orchestrator owns lifecycle
    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cartesia_ws_error(fake_settings: Settings) -> None:
    """WebSocket connection error → result.error populated, audio_path None."""
    provider = CartesiaTTSProvider(fake_settings, model="sonic-3", voice="test-voice-id")

    with patch(
        "coval_bench.providers.tts.cartesia.AsyncCartesia",
        side_effect=RuntimeError("websocket refused"),
    ):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "websocket refused" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_cartesia_no_audio_chunks(fake_settings: Settings) -> None:
    """No audio chunks received → audio_path is None, error is None."""
    provider = CartesiaTTSProvider(fake_settings, model="sonic-3", voice="test-voice-id")
    fake_client = make_fake_cartesia_client([])  # empty chunks

    with patch("coval_bench.providers.tts.cartesia.AsyncCartesia", return_value=fake_client):
        result = await provider.synthesize("silence test")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_cartesia_name_and_model(fake_settings: Settings) -> None:
    p = CartesiaTTSProvider(fake_settings, model="sonic-turbo", voice="voice-id")
    assert p.name == "cartesia-sonic-turbo"
    assert p.model == "sonic-turbo"


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_cartesia_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",  # type: ignore[arg-type]
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        cartesia_api_key=None,
    )
    with pytest.raises(ValueError, match="cartesia_api_key"):
        CartesiaTTSProvider(settings_no_key, model="sonic-3", voice="test-voice")
