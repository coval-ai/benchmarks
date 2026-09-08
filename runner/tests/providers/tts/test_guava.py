# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Guava (daytona-tts) TTS provider."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers import _http_session
from coval_bench.providers.tts.guava import GuavaTTSProvider

from .conftest import make_pcm_bytes

_BASE_URL = "https://guava.example.internal"
_VOICE = "grace"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql://runner:password@localhost:5432/benchmarks",
        "dataset_bucket": "test-bucket",
        "dataset_id": "stt-v1",
        "log_level": "DEBUG",
        "guava_base_url": _BASE_URL,
        "guava_api_key": SecretStr("test-guava-key"),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _install_mock(handler: Any) -> None:
    _http_session._CLIENTS["guava"] = httpx.AsyncClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture(autouse=True)
def reset_clients() -> Generator[None, None, None]:
    _http_session._CLIENTS.clear()
    yield
    _http_session._CLIENTS.clear()


@pytest.mark.asyncio
async def test_guava_happy_path() -> None:
    pcm = make_pcm_bytes(480) * 4
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, content=pcm)

    _install_mock(handler)

    provider = GuavaTTSProvider(_settings(), model="daytona-tts", voice=_VOICE)
    result = await provider.synthesize("Hello from Guava")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 10_000
    assert result.audio_path is not None and result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "guava"
    assert result.model == "daytona-tts"

    assert str(captured["url"]).endswith("/audio/speech")
    assert captured["authorization"] == "Bearer test-guava-key"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["input"] == "Hello from Guava"
    assert body["voice"] == _VOICE
    assert body["sampling_rate"] == 16000
    assert body["response_format"] == "wav"
    assert body["stream"] is True

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_guava_http_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b'{"detail": "rate limit"}')

    _install_mock(handler)

    provider = GuavaTTSProvider(_settings(), model="daytona-tts", voice=_VOICE)
    result = await provider.synthesize("hi")

    assert result.error is not None
    assert "429" in result.error
    assert "rate limit" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_guava_transport_exception() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic network failure")

    _install_mock(handler)

    provider = GuavaTTSProvider(_settings(), model="daytona-tts", voice=_VOICE)
    result = await provider.synthesize("hi")

    assert result.error is not None
    assert "synthetic network failure" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_guava_empty_response() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    _install_mock(handler)

    provider = GuavaTTSProvider(_settings(), model="daytona-tts", voice=_VOICE)
    result = await provider.synthesize("silence")

    assert result.error == ("provider closed the stream without sending audio or an error")
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_guava_warmup_synthesizes_and_cleans_up() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(200, content=make_pcm_bytes(240))

    _install_mock(handler)
    await GuavaTTSProvider.warmup(_settings())

    assert seen == ["POST /audio/speech"]


@pytest.mark.asyncio
async def test_guava_warmup_noop_without_config() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=make_pcm_bytes(240))

    _install_mock(handler)
    # No URL / key configured → warmup must return without touching the network.
    await GuavaTTSProvider.warmup(_settings(guava_base_url=None, guava_api_key=None))

    assert calls == []


def test_guava_name_and_model() -> None:
    p = GuavaTTSProvider(_settings(), model="daytona-tts", voice=_VOICE)
    assert p.name == "guava-daytona-tts"
    assert p.model == "daytona-tts"


def test_guava_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported Guava model"):
        GuavaTTSProvider(_settings(), model="nonexistent", voice=_VOICE)


def test_guava_missing_base_url() -> None:
    with pytest.raises(ValueError, match="guava_base_url"):
        GuavaTTSProvider(_settings(guava_base_url=None), model="daytona-tts", voice=_VOICE)


def test_guava_missing_api_key() -> None:
    with pytest.raises(ValueError, match="guava_api_key"):
        GuavaTTSProvider(_settings(guava_api_key=None), model="daytona-tts", voice=_VOICE)
