# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ElevenLabs TTS provider."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from coval_bench.config import Settings
from coval_bench.providers import _http_session
from coval_bench.providers.tts.elevenlabs import ElevenLabsTTSProvider

from .conftest import make_pcm_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_mock(handler: object) -> None:
    """Register a MockTransport-backed AsyncClient as the shared elevenlabs client."""
    _http_session._CLIENTS["elevenlabs"] = httpx.AsyncClient(
        base_url="https://api.elevenlabs.io",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def reset_clients() -> None:
    _http_session._CLIENTS.clear()
    yield
    _http_session._CLIENTS.clear()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_happy_path(
    fake_settings: Settings,
    tmp_path: Path,
) -> None:
    pcm = make_pcm_bytes(480) * 4  # ~80 ms of silence
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["xi_api_key"] = request.headers.get("xi-api-key")
        captured["body"] = request.read()
        return httpx.Response(200, content=pcm)

    _install_mock(handler)

    provider = ElevenLabsTTSProvider(
        fake_settings,
        model="eleven_flash_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
    )
    result = await provider.synthesize("Hello from ElevenLabs")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 10_000
    assert result.audio_path is not None and result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "elevenlabs"
    assert result.model == "eleven_flash_v2_5"
    assert result.http_version == "HTTP/1.1"
    assert result.submit_to_headers_ms is None

    url = str(captured["url"])
    assert "/v1/text-to-speech/IKne3meq5aSn9XLyUdCD/stream" in url
    assert "output_format=pcm_24000" in url
    assert captured["xi_api_key"] == "test-elevenlabs-key"
    body = captured["body"]
    assert isinstance(body, bytes)
    assert b"Hello from ElevenLabs" in body
    assert b"eleven_flash_v2_5" in body

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_elevenlabs_all_models(fake_settings: Settings) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=make_pcm_bytes(240))

    _install_mock(handler)

    for model in ElevenLabsTTSProvider._VALID_MODELS:
        provider = ElevenLabsTTSProvider(fake_settings, model=model, voice="test-voice")
        result = await provider.synthesize("test")
        assert result.error is None, f"{model}: {result.error}"
        if result.audio_path is not None:
            result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_http_error(fake_settings: Settings) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b'{"detail": "rate limit"}')

    _install_mock(handler)

    provider = ElevenLabsTTSProvider(fake_settings, model="eleven_flash_v2_5", voice="v")
    result = await provider.synthesize("hi")

    assert result.error is not None
    assert "429" in result.error
    assert "rate limit" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_elevenlabs_transport_exception(fake_settings: Settings) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic network failure")

    _install_mock(handler)

    provider = ElevenLabsTTSProvider(fake_settings, model="eleven_flash_v2_5", voice="v")
    result = await provider.synthesize("hi")

    assert result.error is not None
    assert "synthetic network failure" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_elevenlabs_empty_response(fake_settings: Settings) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    _install_mock(handler)

    provider = ElevenLabsTTSProvider(fake_settings, model="eleven_flash_v2_5", voice="v")
    result = await provider.synthesize("silence")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# warmup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_warmup_issues_head(fake_settings: Settings) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(401, content=b"unauthorized")

    _install_mock(handler)
    await ElevenLabsTTSProvider.warmup(fake_settings)

    assert seen == ["HEAD /v1/voices"]


@pytest.mark.asyncio
async def test_elevenlabs_warmup_propagates_transport_error(fake_settings: Settings) -> None:
    """Transport failures propagate so the orchestrator can log them."""

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns blew up")

    _install_mock(handler)
    with pytest.raises(httpx.ConnectError):
        await ElevenLabsTTSProvider.warmup(fake_settings)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_elevenlabs_name_and_model(fake_settings: Settings) -> None:
    p = ElevenLabsTTSProvider(fake_settings, model="eleven_turbo_v2_5", voice="voice-id")
    assert p.name == "elevenlabs-eleven_turbo_v2_5"
    assert p.model == "eleven_turbo_v2_5"


def test_elevenlabs_rejects_unsupported_model(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unsupported ElevenLabs model"):
        ElevenLabsTTSProvider(fake_settings, model="not-a-real-model", voice="v")


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
