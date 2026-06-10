# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenAI TTS provider (HTTP + Realtime paths)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from coval_bench.config import Settings
from coval_bench.providers import _http_session
from coval_bench.providers.tts.openai import HTTP_MODELS, VALID_VOICES, OpenAITTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes


@pytest.fixture(autouse=True)
def _reset_http_clients() -> None:
    _http_session._CLIENTS.clear()
    yield
    _http_session._CLIENTS.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_http_provider(
    fake_settings: Settings, model: str = "gpt-4o-mini-tts"
) -> OpenAITTSProvider:
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
    mock_resp.http_version = "HTTP/2"
    mock_resp.http_response.request.extensions = {"__t_submit": 1.0, "__t_headers": 1.002}
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
    assert result.model == "gpt-4o-mini-tts"
    assert result.voice == "alloy"
    assert result.http_version == "HTTP/2"
    assert result.submit_to_headers_ms == 2.0

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
    provider = OpenAITTSProvider(fake_settings, model="gpt-4o-mini-tts", voice="invalid_voice")
    assert provider._voice == "alloy"


def test_openai_valid_voices_accepted(fake_settings: Settings) -> None:
    """All documented voices are accepted without fallback."""
    for voice in VALID_VOICES:
        p = OpenAITTSProvider(fake_settings, model="gpt-4o-mini-tts", voice=voice)
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

    with patch(
        "coval_bench.providers.tts.openai.websockets.connect",
        return_value=fake_ws,
    ):
        result = await provider.synthesize("Hello realtime")

    assert result.error is None, f"Error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0

    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# name / model properties
# ---------------------------------------------------------------------------


def test_openai_name_property(fake_settings: Settings) -> None:
    p = OpenAITTSProvider(fake_settings, model="gpt-4o-mini-tts", voice="echo")
    assert p.name == "openai-gpt-4o-mini-tts"
    assert p.model == "gpt-4o-mini-tts"


# ---------------------------------------------------------------------------
# gpt-4o-mini-tts is the only OpenAI HTTP model in production.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared client injection + warmup
# ---------------------------------------------------------------------------


def test_openai_uses_shared_http_client(fake_settings: Settings) -> None:
    """Construction registers a pooled httpx client in the shared registry."""
    OpenAITTSProvider(fake_settings, model="gpt-4o-mini-tts", voice="alloy")
    assert "openai" in _http_session._CLIENTS
    assert isinstance(_http_session._CLIENTS["openai"], httpx.AsyncClient)


@pytest.mark.asyncio
async def test_openai_warmup_issues_head(fake_settings: Settings) -> None:
    """warmup() HEADs /v1/models on the shared client; a 401 does not raise."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(401, content=b"unauthorized")

    _http_session._CLIENTS["openai"] = httpx.AsyncClient(
        base_url="https://api.openai.com",
        transport=httpx.MockTransport(handler),
    )
    await OpenAITTSProvider.warmup(fake_settings)
    assert seen == ["HEAD /v1/models"]


@pytest.mark.asyncio
async def test_openai_warmup_propagates_transport_error(fake_settings: Settings) -> None:
    """Transport failures propagate so the orchestrator can log them."""

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns blew up")

    _http_session._CLIENTS["openai"] = httpx.AsyncClient(
        base_url="https://api.openai.com",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(httpx.ConnectError):
        await OpenAITTSProvider.warmup(fake_settings)


@pytest.mark.asyncio
async def test_synthesize_gpt_4o_mini_tts_streams_pcm(fake_settings: Settings) -> None:
    """gpt-4o-mini-tts streams PCM chunks → ttfa set, valid WAV with .wav magic bytes."""
    pcm = make_pcm_bytes(480)
    provider = _make_http_provider(fake_settings, model="gpt-4o-mini-tts")

    mock_client = _make_streaming_response_mock([pcm, pcm])
    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini-tts"
    assert result.voice == "alloy"
    assert result.audio_path is not None
    assert result.audio_path.exists()
    # WAV magic bytes — RIFF header
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    result.audio_path.unlink()
