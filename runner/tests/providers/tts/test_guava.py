# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Guava (daytona-tts) TTS provider."""

from __future__ import annotations

import io
import json
import wave
from collections.abc import AsyncIterator, Generator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers import _http_session
from coval_bench.providers.tts.guava import GuavaTTSProvider

from .conftest import make_pcm_bytes

_BASE_URL = "https://guava.example.internal"
_VOICE = "grace"


def _wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


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
        return httpx.Response(200, content=_wav(pcm))

    _install_mock(handler)

    provider = GuavaTTSProvider(_settings(), model="daytona-tts", voice=_VOICE)
    result = await provider.synthesize("Hello from Guava")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 10_000
    assert result.audio_path is not None and result.audio_path.exists()
    with wave.open(str(result.audio_path), "rb") as wav:
        assert wav.readframes(wav.getnframes()) == pcm
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
        return httpx.Response(200, content=_wav(make_pcm_bytes(240)))

    _install_mock(handler)
    await GuavaTTSProvider.warmup(_settings())

    assert seen == ["POST /audio/speech"]


@pytest.mark.asyncio
async def test_guava_warmup_noop_without_config() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=_wav(make_pcm_bytes(240)))

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


@pytest.mark.asyncio
async def test_guava_silent_wav_is_not_a_success() -> None:
    _install_mock(lambda _: httpx.Response(200, content=_wav(bytes(32000))))
    result = await GuavaTTSProvider(_settings(), "daytona-tts", _VOICE).synthesize("hi")
    assert result.error == "provider audio remained below the audibility threshold"
    assert result.ttfa_ms is None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_guava_header_arrival_does_not_start_audio_clock() -> None:
    data = _wav(make_pcm_bytes(480))

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield data[:20]
            yield data[20:44]
            yield data[44:]

    _install_mock(lambda _: httpx.Response(200, stream=Stream()))
    with patch("coval_bench.providers.tts.guava.time") as clock:
        clock.monotonic.side_effect = [100.0, 100.1, 100.2, 103.0]
        result = await GuavaTTSProvider(_settings(), "daytona-tts", _VOICE).synthesize("hi")
    assert result.error is None
    assert result.ttfa_ms == pytest.approx(3000.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [b"not a WAV", _wav(bytes(320), sample_rate=8000)])
async def test_guava_rejects_invalid_wav(data: bytes) -> None:
    _install_mock(lambda _: httpx.Response(200, content=data))
    result = await GuavaTTSProvider(_settings(), "daytona-tts", _VOICE).synthesize("hi")
    assert result.error is not None
    assert result.ttfa_ms is None
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_guava_live_raw_pcm_response_format() -> None:
    pcm = make_pcm_bytes(480)
    _install_mock(
        lambda _: httpx.Response(
            200, content=pcm, headers={"Content-Type": "audio/L16;rate=16000;channels=1"}
        )
    )
    result = await GuavaTTSProvider(_settings(), "daytona-tts", _VOICE).synthesize("hi")
    assert result.error is None
    assert result.audio_path is not None
    with wave.open(str(result.audio_path), "rb") as wav:
        assert wav.readframes(wav.getnframes()) == pcm
    result.audio_path.unlink()
