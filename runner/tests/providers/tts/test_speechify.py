# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Speechify TTS provider."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers import _http_session
from coval_bench.providers.tts.speechify import SpeechifyTTSProvider

from .conftest import make_pcm_bytes

_VOICE = "geffen_32"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql://runner:password@localhost:5432/benchmarks",
        "dataset_bucket": "test-bucket",
        "dataset_id": "stt-v1",
        "runner_sha": "test",
        "log_level": "DEBUG",
        "speechify_api_key": SecretStr("test-speechify-key"),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _install_mock(handler: Any) -> None:
    _http_session._CLIENTS["speechify"] = httpx.AsyncClient(
        base_url="https://api.speechify.ai",
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture(autouse=True)
def reset_clients() -> Generator[None, None, None]:
    _http_session._CLIENTS.clear()
    yield
    _http_session._CLIENTS.clear()


@pytest.mark.asyncio
async def test_speechify_happy_path() -> None:
    pcm = make_pcm_bytes(480) * 4
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["accept"] = request.headers.get("accept")
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, content=pcm)

    _install_mock(handler)

    provider = SpeechifyTTSProvider(_settings(), model="simba-3.2", voice=_VOICE)
    result = await provider.synthesize("Hello from Speechify")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 10_000
    assert result.audio_path is not None and result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "speechify"
    assert result.model == "simba-3.2"

    assert str(captured["url"]).endswith("/v1/audio/stream")
    assert captured["authorization"] == "Bearer test-speechify-key"
    assert captured["accept"] == "audio/pcm"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["input"] == "Hello from Speechify"
    assert body["voice_id"] == _VOICE
    assert body["model"] == "simba-3.2"
    assert body["output_format"] == "pcm_24000"

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_speechify_all_models() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=make_pcm_bytes(240))

    _install_mock(handler)

    for model in SpeechifyTTSProvider._VALID_MODELS:
        provider = SpeechifyTTSProvider(_settings(), model=model, voice=_VOICE)
        result = await provider.synthesize("test")
        assert result.error is None, f"{model}: {result.error}"
        if result.audio_path is not None:
            result.audio_path.unlink()


@pytest.mark.asyncio
async def test_speechify_http_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b'{"detail": "rate limit"}')

    _install_mock(handler)

    provider = SpeechifyTTSProvider(_settings(), model="simba-3.2", voice=_VOICE)
    result = await provider.synthesize("hi")

    assert result.error is not None
    assert "429" in result.error
    assert "rate limit" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_speechify_transport_exception() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic network failure")

    _install_mock(handler)

    provider = SpeechifyTTSProvider(_settings(), model="simba-3.2", voice=_VOICE)
    result = await provider.synthesize("hi")

    assert result.error is not None
    assert "synthetic network failure" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_speechify_empty_response() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    _install_mock(handler)

    provider = SpeechifyTTSProvider(_settings(), model="simba-3.2", voice=_VOICE)
    result = await provider.synthesize("silence")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_speechify_warmup_issues_head() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(401, content=b"unauthorized")

    _install_mock(handler)
    await SpeechifyTTSProvider.warmup(_settings())

    assert seen == ["HEAD /v1/voices"]


@pytest.mark.asyncio
async def test_speechify_warmup_propagates_transport_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns blew up")

    _install_mock(handler)
    with pytest.raises(httpx.ConnectError):
        await SpeechifyTTSProvider.warmup(_settings())


def test_speechify_name_and_model() -> None:
    p = SpeechifyTTSProvider(_settings(), model="simba-3.0", voice=_VOICE)
    assert p.name == "speechify-simba-3.0"
    assert p.model == "simba-3.0"


def test_speechify_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported Speechify model"):
        SpeechifyTTSProvider(_settings(), model="simba-english", voice=_VOICE)


def test_speechify_missing_api_key() -> None:
    with pytest.raises(ValueError, match="speechify_api_key"):
        SpeechifyTTSProvider(_settings(speechify_api_key=None), model="simba-3.2", voice=_VOICE)
