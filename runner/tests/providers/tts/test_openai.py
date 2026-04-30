# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenAI TTS provider (HTTP + Realtime paths)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.openai import HTTP_MODELS, VALID_VOICES, OpenAITTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_http_provider(fake_settings: Settings, model: str = "tts-1") -> OpenAITTSProvider:
    return OpenAITTSProvider(fake_settings, model=model, voice="alloy")


def _make_realtime_provider(fake_settings: Settings) -> OpenAITTSProvider:
    return OpenAITTSProvider(fake_settings, model="gpt-realtime-2025-08-28", voice="alloy")


def _make_streaming_response_mock(chunks: list[bytes]) -> MagicMock:
    """Return a mock for client.audio.speech.with_streaming_response.create(...)."""

    async def _iter_bytes() -> Any:
        for c in chunks:
            yield c

    mock_resp = MagicMock()
    mock_resp.iter_bytes = _iter_bytes
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_with_streaming = MagicMock()
    mock_with_streaming.create = MagicMock(return_value=mock_resp)

    mock_audio = MagicMock()
    mock_audio.speech.with_streaming_response = mock_with_streaming

    mock_client = MagicMock()
    mock_client.audio = mock_audio
    return mock_client


# ---------------------------------------------------------------------------
# HTTP path — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_http_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    """HTTP streaming synthesize → ttfa set, valid WAV written."""
    pcm = make_pcm_bytes()
    provider = _make_http_provider(fake_settings)

    mock_client = _make_streaming_response_mock([pcm])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "openai"
    assert result.model == "tts-1"
    assert result.voice == "alloy"

    # Orchestrator owns deletion — provider must NOT auto-delete
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_openai_http_all_http_models(fake_settings: Settings) -> None:
    """All HTTP models go through _synthesize_http successfully."""
    pcm = make_pcm_bytes()
    for model in HTTP_MODELS:
        provider = OpenAITTSProvider(fake_settings, model=model, voice="alloy")
        mock_client = _make_streaming_response_mock([pcm])
        with patch.object(provider, "_client", mock_client):
            result = await provider.synthesize("test")
        assert result.error is None, f"Model {model} failed: {result.error}"
        if result.audio_path:
            result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Bad voice — fallback to alloy
# ---------------------------------------------------------------------------


def test_openai_unknown_voice_falls_back_to_alloy(fake_settings: Settings) -> None:
    """Unknown voice is replaced by 'alloy' at construction time."""
    provider = OpenAITTSProvider(fake_settings, model="tts-1", voice="invalid_voice")
    assert provider._voice == "alloy"


def test_openai_valid_voices_accepted(fake_settings: Settings) -> None:
    """All documented voices are accepted without fallback."""
    for voice in VALID_VOICES:
        p = OpenAITTSProvider(fake_settings, model="tts-1", voice=voice)
        assert p._voice == voice


# ---------------------------------------------------------------------------
# Error path — HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_http_error_path(fake_settings: Settings) -> None:
    """Streaming exception → result.error populated, audio_path is None."""
    provider = _make_http_provider(fake_settings)

    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(side_effect=RuntimeError("connection refused"))
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_with_streaming = MagicMock()
    mock_with_streaming.create = MagicMock(return_value=mock_resp)
    mock_audio = MagicMock()
    mock_audio.speech.with_streaming_response = mock_with_streaming
    mock_client = MagicMock()
    mock_client.audio = mock_audio

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.audio_path is None


# ---------------------------------------------------------------------------
# Unsupported model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_unsupported_model(fake_settings: Settings) -> None:
    """Unknown model returns error result without calling any network."""
    provider = OpenAITTSProvider(fake_settings, model="gpt-unsupported", voice="alloy")
    result = await provider.synthesize("test")
    assert result.error is not None
    assert "Unsupported" in result.error
    assert result.audio_path is None


# ---------------------------------------------------------------------------
# Realtime path — happy path
# ---------------------------------------------------------------------------


def _make_realtime_ws_messages(audio_b64: str) -> list[str]:
    """Build the sequence of WS messages the realtime API would send."""
    return [
        json.dumps({"type": "response.audio.delta", "delta": audio_b64}),
        json.dumps({"type": "response.done"}),
    ]


@pytest.mark.asyncio
async def test_openai_realtime_happy_path(fake_settings: Settings) -> None:
    """Realtime WS synthesize → ttfa set, audio file written."""
    pcm = make_pcm_bytes(480)
    audio_b64 = base64.b64encode(pcm).decode()
    messages = _make_realtime_ws_messages(audio_b64)
    fake_ws = FakeWebSocket(messages)

    provider = _make_realtime_provider(fake_settings)

    # Mock the aiohttp session POST (session creation)
    mock_sess_resp = MagicMock()
    mock_sess_resp.status = 200
    mock_sess_resp.__aenter__ = AsyncMock(return_value=mock_sess_resp)
    mock_sess_resp.__aexit__ = AsyncMock(return_value=False)

    mock_http_sess = MagicMock()
    mock_http_sess.post = MagicMock(return_value=mock_sess_resp)
    mock_http_sess.__aenter__ = AsyncMock(return_value=mock_http_sess)
    mock_http_sess.__aexit__ = AsyncMock(return_value=False)

    aiohttp_patch = patch(
        "coval_bench.providers.tts.openai.aiohttp.ClientSession",
        return_value=mock_http_sess,
    )
    ws_patch = patch(
        "coval_bench.providers.tts.openai.websockets.connect",
        return_value=fake_ws,
    )
    with aiohttp_patch, ws_patch:
        result = await provider.synthesize("Hello realtime")

    assert result.error is None, f"Error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0

    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Realtime — session creation error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_realtime_session_error(fake_settings: Settings) -> None:
    """Session creation failure → error result, no audio."""
    provider = _make_realtime_provider(fake_settings)

    mock_sess_resp = MagicMock()
    mock_sess_resp.status = 401
    mock_sess_resp.text = AsyncMock(return_value="Unauthorized")
    mock_sess_resp.__aenter__ = AsyncMock(return_value=mock_sess_resp)
    mock_sess_resp.__aexit__ = AsyncMock(return_value=False)

    mock_http_sess = MagicMock()
    mock_http_sess.post = MagicMock(return_value=mock_sess_resp)
    mock_http_sess.__aenter__ = AsyncMock(return_value=mock_http_sess)
    mock_http_sess.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "coval_bench.providers.tts.openai.aiohttp.ClientSession",
        return_value=mock_http_sess,
    ):
        result = await provider.synthesize("test")

    assert result.error is not None
    assert result.audio_path is None


# ---------------------------------------------------------------------------
# name / model properties
# ---------------------------------------------------------------------------


def test_openai_name_property(fake_settings: Settings) -> None:
    p = OpenAITTSProvider(fake_settings, model="tts-1-hd", voice="echo")
    assert p.name == "openai-tts-1-hd"
    assert p.model == "tts-1-hd"


# ---------------------------------------------------------------------------
# Re-activated 2026-04-30: tts-1-hd is the only OpenAI HTTP model in production.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_tts_1_hd_streams_pcm(fake_settings: Settings) -> None:
    """tts-1-hd streams PCM chunks → ttfa set, valid WAV with .wav magic bytes."""
    pcm = make_pcm_bytes(480)
    provider = _make_http_provider(fake_settings, model="tts-1-hd")

    mock_client = _make_streaming_response_mock([pcm, pcm])
    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.provider == "openai"
    assert result.model == "tts-1-hd"
    assert result.voice == "alloy"
    assert result.audio_path is not None
    assert result.audio_path.exists()
    # WAV magic bytes — RIFF header
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    result.audio_path.unlink()
