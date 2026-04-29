# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hume TTS provider.

The ``hume`` package is an optional dependency.  Tests in this file are
skipped automatically when the extra is not installed::

    pytest.importorskip("hume")

To run these tests locally::

    pip install "coval-bench[hume-tts]"
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

hume = pytest.importorskip("hume")

from coval_bench.config import Settings  # noqa: E402
from coval_bench.providers.tts.hume import HumeTTSProvider  # noqa: E402

from .conftest import make_wav_bytes  # noqa: E402

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hume_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    """Hume octave-tts synthesize → ttfa set, WAV file written."""
    wav_chunks = [make_wav_bytes(240)]
    provider = HumeTTSProvider(
        fake_settings, model="octave-tts", voice="176a55b1-4468-4736-8878-db82729667c1"
    )

    mock_tts = MagicMock()
    mock_tts.synthesize_file_streaming.return_value = iter(wav_chunks)

    mock_client = MagicMock()
    mock_client.tts = mock_tts

    with patch("coval_bench.providers.tts.hume.HumeClient", return_value=mock_client):
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
    wav_chunk = make_wav_bytes(240)
    provider = HumeTTSProvider(fake_settings, model="octave-2", voice="test-voice-id")

    mock_tts = MagicMock()
    mock_tts.synthesize_file_streaming.return_value = iter([wav_chunk])
    mock_client = MagicMock()
    mock_client.tts = mock_tts

    with patch("coval_bench.providers.tts.hume.HumeClient", return_value=mock_client):
        result = await provider.synthesize("test octave-2")

    assert result.error is None
    if result.audio_path:
        result.audio_path.unlink()


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
    """SDK exception → result.error populated, audio_path None."""
    provider = HumeTTSProvider(fake_settings, model="octave-tts", voice="test-voice-id")

    with patch(
        "coval_bench.providers.tts.hume.HumeClient",
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

    mock_tts = MagicMock()
    mock_tts.synthesize_file_streaming.return_value = iter([])
    mock_client = MagicMock()
    mock_client.tts = mock_tts

    with patch("coval_bench.providers.tts.hume.HumeClient", return_value=mock_client):
        result = await provider.synthesize("silence")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


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
        database_url="postgresql://runner:password@localhost:5432/benchmarks",  # type: ignore[arg-type]
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        hume_api_key=None,
    )
    with pytest.raises(ValueError, match="hume_api_key"):
        HumeTTSProvider(settings_no_key, model="octave-tts", voice="v")
