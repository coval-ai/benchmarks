# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AssemblyAI TTS provider (OpenAI-compatible HTTP streaming)."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import NOT_GIVEN

from coval_bench.config import Settings
from coval_bench.providers import _http_session
from coval_bench.providers.tts.assemblyai import VALID_VOICES, AssemblyAITTSProvider

from .conftest import make_pcm_bytes

_MODEL = "assemblyai-tts-1"
_CLONED_VOICE = "01a1604d-6fb3-7641-ad62-50b4cd8729db"


@pytest.fixture(autouse=True)
def _reset_http_clients() -> Generator[None, None, None]:
    _http_session._CLIENTS.clear()
    yield
    _http_session._CLIENTS.clear()


def _make_provider(fake_settings: Settings, voice: str = "aai-audrey") -> AssemblyAITTSProvider:
    return AssemblyAITTSProvider(fake_settings, model=_MODEL, voice=voice)


def _make_streaming_response_mock(chunks: list[bytes]) -> MagicMock:
    async def _iter_bytes() -> Any:
        for c in chunks:
            yield c

    mock_resp = MagicMock()
    mock_resp.iter_bytes = _iter_bytes
    mock_resp.http_version = "HTTP/2"
    mock_resp.http_response.request.extensions = {"__t_submit": 1.0, "__t_headers": 1.002}
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.audio.speech.with_streaming_response.create = MagicMock(return_value=mock_resp)
    return mock_client


@pytest.mark.asyncio
async def test_assemblyai_tts_happy_path(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings)
    mock_client = _make_streaming_response_mock([make_pcm_bytes(480), make_pcm_bytes(480)])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello from AssemblyAI")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.provider == "assemblyai"
    assert result.model == _MODEL
    assert result.voice == "aai-audrey"
    assert result.http_version == "HTTP/2"
    assert result.submit_to_headers_ms == 2.0
    assert result.audio_path is not None
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_assemblyai_tts_sends_voice_id_without_voice(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings, voice=_CLONED_VOICE)
    mock_client = _make_streaming_response_mock([make_pcm_bytes(480)])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello")

    assert result.error is None, result.error
    kwargs = mock_client.audio.speech.with_streaming_response.create.call_args.kwargs
    assert kwargs["model"] == _MODEL
    assert kwargs["voice"] is NOT_GIVEN
    assert kwargs["response_format"] == "pcm"
    assert kwargs["extra_body"] == {"voice_id": _CLONED_VOICE}
    if result.audio_path:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_assemblyai_tts_error_path(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings)
    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(side_effect=RuntimeError("connection refused"))
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_client = MagicMock()
    mock_client.audio.speech.with_streaming_response.create = MagicMock(return_value=mock_resp)

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.audio_path is None


def test_assemblyai_tts_client_targets_openai_prefix(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings)
    assert str(provider._client.base_url) == "https://tts.us.assemblyai.com/v1/openai/"
    assert "assemblyai" in _http_session._CLIENTS
    assert isinstance(_http_session._CLIENTS["assemblyai"], httpx.AsyncClient)


@pytest.mark.parametrize("voice", [*VALID_VOICES, _CLONED_VOICE])
def test_assemblyai_tts_accepts_slugs_and_uuids(fake_settings: Settings, voice: str) -> None:
    provider = _make_provider(fake_settings, voice=voice)
    assert provider.name == f"assemblyai-{_MODEL}"
    assert provider.model == _MODEL


def test_assemblyai_tts_rejects_unknown_voice(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid AssemblyAI TTS voice"):
        _make_provider(fake_settings, voice="alloy")


def test_assemblyai_tts_rejects_unknown_model(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid AssemblyAI TTS model"):
        AssemblyAITTSProvider(fake_settings, model="universal-3.6-pro", voice="aai-audrey")


def test_assemblyai_tts_missing_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        log_level="DEBUG",
        assemblyai_api_key=None,
    )
    with pytest.raises(ValueError, match="assemblyai_api_key is required"):
        AssemblyAITTSProvider(settings, model=_MODEL, voice="aai-audrey")


@pytest.mark.asyncio
async def test_assemblyai_tts_warmup_heads_speech_endpoint(fake_settings: Settings) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(401, content=b"unauthorized")

    _http_session._CLIENTS["assemblyai"] = httpx.AsyncClient(
        base_url="https://tts.us.assemblyai.com",
        transport=httpx.MockTransport(handler),
    )
    await AssemblyAITTSProvider.warmup(fake_settings)
    assert seen == ["HEAD /v1/openai/audio/speech"]
