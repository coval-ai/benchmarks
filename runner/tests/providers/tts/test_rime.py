# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rime TTS provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.rime import RimeTTSProvider

from .conftest import FakeAiohttpResponse, FakeAiohttpSession, make_pcm_bytes

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    """Rime HTTP streaming → ttfa set, WAV written."""
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240)]
    provider = RimeTTSProvider(fake_settings, model="arcana", voice="luna")

    fake_resp = FakeAiohttpResponse(chunks, status=200)
    fake_sess = FakeAiohttpSession(fake_resp)

    with patch("coval_bench.providers.tts.rime.aiohttp.ClientSession", return_value=fake_sess):
        result = await provider.synthesize("Hello from Rime")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "rime"
    assert result.model == "arcana"
    assert result.voice == "luna"

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_rime_mistv2_model(fake_settings: Settings) -> None:
    """mistv2 model also synthesizes successfully."""
    chunks = [make_pcm_bytes(240)]
    provider = RimeTTSProvider(fake_settings, model="mistv2", voice="luna")

    fake_resp = FakeAiohttpResponse(chunks, status=200)
    fake_sess = FakeAiohttpSession(fake_resp)

    with patch("coval_bench.providers.tts.rime.aiohttp.ClientSession", return_value=fake_sess):
        result = await provider.synthesize("mistv2 test")

    assert result.error is None
    if result.audio_path:
        result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Bad model raises error immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_invalid_model(fake_settings: Settings) -> None:
    """Unsupported model returns error result without calling network."""
    provider = RimeTTSProvider(fake_settings, model="unknown-model", voice="luna")
    result = await provider.synthesize("test")
    assert result.error is not None
    assert "Unsupported" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# Error path — HTTP errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_http_error(fake_settings: Settings) -> None:
    """Non-200 response → result.error populated."""
    provider = RimeTTSProvider(fake_settings, model="arcana", voice="luna")

    fake_resp = FakeAiohttpResponse([], status=429, text_body="Too Many Requests")
    fake_sess = FakeAiohttpSession(fake_resp)

    with patch("coval_bench.providers.tts.rime.aiohttp.ClientSession", return_value=fake_sess):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "429" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_rime_network_exception(fake_settings: Settings) -> None:
    """Network exception → result.error populated, audio_path None."""
    provider = RimeTTSProvider(fake_settings, model="arcana", voice="luna")

    with patch(
        "coval_bench.providers.tts.rime.aiohttp.ClientSession",
        side_effect=RuntimeError("connection reset"),
    ):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert result.audio_path is None


# ---------------------------------------------------------------------------
# No audio chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_empty_response(fake_settings: Settings) -> None:
    """Empty body → audio_path None, error None."""
    provider = RimeTTSProvider(fake_settings, model="arcana", voice="luna")

    fake_resp = FakeAiohttpResponse([], status=200)
    fake_sess = FakeAiohttpSession(fake_resp)

    with patch("coval_bench.providers.tts.rime.aiohttp.ClientSession", return_value=fake_sess):
        result = await provider.synthesize("silent test")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_rime_name_and_model(fake_settings: Settings) -> None:
    p = RimeTTSProvider(fake_settings, model="arcana", voice="luna")
    assert p.name == "rime-arcana"
    assert p.model == "arcana"


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_rime_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",  # type: ignore[arg-type]
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        rime_api_key=None,
    )
    with pytest.raises(ValueError, match="rime_api_key"):
        RimeTTSProvider(settings_no_key, model="arcana", voice="luna")
