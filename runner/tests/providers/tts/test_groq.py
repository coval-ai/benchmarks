# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from coval_bench.config import Settings
from coval_bench.providers import _http_session
from coval_bench.providers.tts.groq import (
    _MAX_INPUT_CHARS,
    MODEL_ID,
    VALID_VOICES,
    GroqTTSProvider,
)

from .conftest import make_wav_bytes


@pytest.fixture(autouse=True)
def _reset_http_clients() -> Generator[None, None, None]:
    _http_session._CLIENTS.clear()
    yield
    _http_session._CLIENTS.clear()


def _make_provider(fake_settings: Settings, voice: str = "autumn") -> GroqTTSProvider:
    return GroqTTSProvider(fake_settings, model=MODEL_ID, voice=voice)


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

    mock_with_streaming = MagicMock()
    mock_with_streaming.create = MagicMock(return_value=mock_resp)

    mock_audio = MagicMock()
    mock_audio.speech.with_streaming_response = mock_with_streaming

    mock_client = MagicMock()
    mock_client.audio = mock_audio
    return mock_client


@pytest.mark.asyncio
async def test_groq_http_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    provider = _make_provider(fake_settings)
    mock_client = _make_streaming_response_mock([make_wav_bytes(480)])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "groq"
    assert result.model == MODEL_ID
    assert result.voice == "autumn"
    assert result.http_version == "HTTP/2"
    assert result.submit_to_headers_ms == 2.0

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_groq_request_uses_wav_format(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings, voice="daniel")
    mock_client = _make_streaming_response_mock([make_wav_bytes(240)])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello from Orpheus")

    create = mock_client.audio.speech.with_streaming_response.create
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == MODEL_ID
    assert kwargs["voice"] == "daniel"
    assert kwargs["input"] == "Hello from Orpheus"
    assert kwargs["response_format"] == "wav"
    if result.audio_path:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_groq_reads_sample_rate_from_wav_header(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings)
    mock_client = _make_streaming_response_mock([make_wav_bytes(480, sample_rate=16000)])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello world")

    assert result.error is None
    assert result.audio_path is not None
    import wave

    with wave.open(str(result.audio_path), "rb") as wf:
        assert wf.getframerate() == 16000
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_groq_non_wav_body_is_error(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings)
    mock_client = _make_streaming_response_mock([b'{"error":"bad request"}'])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "non-WAV" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_groq_empty_audio_is_error(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings)
    mock_client = _make_streaming_response_mock([])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_groq_non_mono_16bit_wav_is_error(fake_settings: Settings) -> None:
    import wave
    from io import BytesIO

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00" * 960)
    provider = _make_provider(fake_settings)
    mock_client = _make_streaming_response_mock([buf.getvalue()])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "mono 16-bit" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_groq_over_length_input_is_error_without_request(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings)
    mock_client = _make_streaming_response_mock([make_wav_bytes(480)])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("x" * (_MAX_INPUT_CHARS + 1))

    assert result.error is not None
    assert str(_MAX_INPUT_CHARS) in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None
    mock_client.audio.speech.with_streaming_response.create.assert_not_called()


@pytest.mark.asyncio
async def test_groq_max_length_input_is_synthesized(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings)
    mock_client = _make_streaming_response_mock([make_wav_bytes(480)])

    with patch.object(provider, "_client", mock_client):
        result = await provider.synthesize("x" * _MAX_INPUT_CHARS)

    assert result.error is None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_groq_http_error_path(fake_settings: Settings) -> None:
    provider = _make_provider(fake_settings)

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


def test_groq_unknown_voice_falls_back(fake_settings: Settings) -> None:
    provider = GroqTTSProvider(fake_settings, model=MODEL_ID, voice="not-a-voice")
    assert provider._voice == "autumn"


def test_groq_valid_voices_accepted(fake_settings: Settings) -> None:
    for voice in VALID_VOICES:
        p = GroqTTSProvider(fake_settings, model=MODEL_ID, voice=voice)
        assert p._voice == voice


def test_groq_invalid_model_raises(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Groq TTS model"):
        GroqTTSProvider(fake_settings, model="whisper-large-v3", voice="autumn")


def test_groq_missing_api_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        groq_api_key=None,
    )
    with pytest.raises(ValueError, match="groq_api_key is required"):
        GroqTTSProvider(settings, model=MODEL_ID, voice="autumn")


def test_groq_name_property(fake_settings: Settings) -> None:
    p = _make_provider(fake_settings)
    assert p.name == f"groq-{MODEL_ID}"
    assert p.model == MODEL_ID


def test_groq_uses_shared_http_client(fake_settings: Settings) -> None:
    _make_provider(fake_settings)
    assert "groq" in _http_session._CLIENTS
    assert isinstance(_http_session._CLIENTS["groq"], httpx.AsyncClient)


@pytest.mark.asyncio
async def test_groq_warmup_issues_head(fake_settings: Settings) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(401, content=b"unauthorized")

    _http_session._CLIENTS["groq"] = httpx.AsyncClient(
        base_url="https://api.groq.com",
        transport=httpx.MockTransport(handler),
    )
    await GroqTTSProvider.warmup(fake_settings)
    assert seen == ["HEAD /openai/v1/models"]


@pytest.mark.asyncio
async def test_groq_warmup_propagates_transport_error(fake_settings: Settings) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns blew up")

    _http_session._CLIENTS["groq"] = httpx.AsyncClient(
        base_url="https://api.groq.com",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(httpx.ConnectError):
        await GroqTTSProvider.warmup(fake_settings)
