# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import wave
from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts import airy

from .conftest import make_pcm_bytes

_VOICE = "a597bb7a98fc9ec1"
_HEADERS = {
    "Content-Type": "audio/pcm",
    "X-Audio-Sample-Rate": "24000",
    "X-Audio-Channels": "1",
    "X-Audio-Sample-Format": "s16le",
}
_PCM = make_pcm_bytes()


class AudioStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, broken: bool = False) -> None:
        self.chunks = chunks
        self.broken = broken
        self.now = 10.0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            self.now = 10.025 if index == 0 else 12.0
            yield chunk
        if self.broken:
            raise httpx.ReadError("stream interrupted")


@pytest.fixture()
def airy_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("AIRY_API_KEY", "test-airy-key")
    settings = Settings(_env_file=None)
    assert settings.airy_api_key is not None
    assert settings.airy_api_key.get_secret_value() == "test-airy-key"
    return settings


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", [_VOICE, "new-custom-voice"])
async def test_stream_request_and_first_chunk_timing_produce_24khz_wav(
    airy_settings: Settings, monkeypatch: pytest.MonkeyPatch, voice: str
) -> None:
    stream = AudioStream([_PCM[:240], _PCM[240:]])

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.airy.so/v1/audio/speech/stream"
        assert request.headers["Authorization"] == "Bearer test-airy-key"
        assert json.loads(request.content) == {
            "model": "airy-tts-v1",
            "input": "Hello from Airy.",
            "voice_id": voice,
            "language": "en",
            "style": "normal",
        }
        request.extensions.update(__t_submit=10.0, __t_headers=10.005, __connection_reused=True)
        return httpx.Response(
            200, headers=_HEADERS, stream=stream, extensions={"http_version": b"HTTP/2"}
        )

    async with httpx.AsyncClient(
        base_url="https://api.airy.so", transport=httpx.MockTransport(handle)
    ) as client:
        monkeypatch.setattr(airy, "get_shared_client", lambda *args: client)
        monkeypatch.setattr(airy, "time", SimpleNamespace(monotonic=lambda: stream.now))
        provider = airy.AiryTTSProvider(airy_settings, model="airy-tts-v1", voice=voice)
        result = await provider.synthesize("Hello from Airy.")

    assert result.error is None
    assert result.provider == "airy"
    assert result.model == "airy-tts-v1"
    assert result.voice == voice
    assert result.ttfa_ms == pytest.approx(25.0)
    assert result.http_version == "HTTP/2"
    assert result.submit_to_headers_ms == pytest.approx(5.0)
    assert result.connection_reused is True
    assert result.audio_path is not None
    try:
        with wave.open(str(result.audio_path), "rb") as wav:
            assert wav.getframerate() == 24000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.readframes(wav.getnframes()) == _PCM
    finally:
        result.audio_path.unlink()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "headers", "chunks", "broken"),
    [
        (401, {"Content-Type": "application/json"}, [b'{"error":"unauthorized"}'], False),
        (429, {"Content-Type": "application/json"}, [b'{"error":"rate limited"}'], False),
        (200, _HEADERS, [], False),
        (200, {**_HEADERS, "X-Audio-Sample-Rate": "48000"}, [_PCM], False),
        (200, {**_HEADERS, "X-Audio-Channels": "2"}, [_PCM], False),
        (200, {**_HEADERS, "X-Audio-Sample-Format": "f32le"}, [_PCM], False),
        (200, {**_HEADERS, "Content-Type": "audio/wav"}, [_PCM], False),
        (200, _HEADERS, [_PCM], True),
    ],
    ids=["auth", "rate-limit", "empty", "sample-rate", "channels", "format", "wav", "broken"],
)
async def test_failed_or_incompatible_stream_never_saves_audio(
    airy_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    headers: dict[str, str],
    chunks: list[bytes],
    broken: bool,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=headers, stream=AudioStream(chunks, broken=broken))

    async with httpx.AsyncClient(
        base_url="https://api.airy.so", transport=httpx.MockTransport(handle)
    ) as client:
        monkeypatch.setattr(airy, "get_shared_client", lambda *args: client)
        provider = airy.AiryTTSProvider(airy_settings, model="airy-tts-v1", voice=_VOICE)
        result = await provider.synthesize("Hello.")

    assert result.error
    assert result.audio_path is None
    if status >= 400:
        assert result.status_code == status


@pytest.mark.parametrize("api_key", [None, SecretStr("")])
def test_missing_api_key_is_rejected(api_key: SecretStr | None) -> None:
    settings = Settings(_env_file=None, airy_api_key=api_key)
    with pytest.raises(ValueError, match="airy_api_key"):
        airy.AiryTTSProvider(settings, model="airy-tts-v1", voice=_VOICE)


@pytest.mark.parametrize("suffix", ["\n", "\x00", "é"])
def test_malformed_key_is_rejected_without_exposing_it(suffix: str) -> None:
    settings = Settings(_env_file=None, airy_api_key=SecretStr("test-private-key" + suffix))
    with pytest.raises(ValueError, match="airy_api_key") as exc:
        airy.AiryTTSProvider(settings, model="airy-tts-v1", voice=_VOICE)
    assert "test-private-key" not in str(exc.value)


def test_invalid_model_is_rejected(airy_settings: Settings) -> None:
    with pytest.raises(ValueError, match="model"):
        airy.AiryTTSProvider(airy_settings, model="unknown-model", voice=_VOICE)


@pytest.mark.asyncio
async def test_warmup_uses_head_without_synthesizing(
    airy_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(405)

    async with httpx.AsyncClient(
        base_url="https://api.airy.so", transport=httpx.MockTransport(handle)
    ) as client:
        monkeypatch.setattr(airy, "get_shared_client", lambda *args: client)
        await airy.AiryTTSProvider.warmup(airy_settings)

    assert len(requests) == 1
    assert requests[0].method == "HEAD"
    assert str(requests[0].url) == "https://api.airy.so/v1/audio/speech/stream"
    assert requests[0].content == b""
