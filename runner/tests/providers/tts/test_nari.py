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
from coval_bench.providers.tts import nari
from coval_bench.registries.provider_keys import PROVIDER_ENV

from .conftest import make_pcm_bytes

_MODEL = "qwen3-tts-fast"
_VOICE = "phoebe"
_PCM_HEADERS = {"Content-Type": "audio/pcm"}
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


@pytest.mark.asyncio
async def test_stream_request_and_first_chunk_timing_produce_24khz_wav(
    fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = AudioStream([_PCM[:240], _PCM[240:]])

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.narilabs.com/v1/audio/speech"
        assert request.headers["Authorization"] == "Bearer test-nari-key"
        assert json.loads(request.content) == {
            "model": _MODEL,
            "input": "Hello from Nari.",
            "voice": _VOICE,
            "language": "en",
            "stream": True,
            "response_format": "pcm",
        }
        request.extensions.update(__t_submit=10.0, __t_headers=10.005, __connection_reused=True)
        return httpx.Response(
            200, headers=_PCM_HEADERS, stream=stream, extensions={"http_version": b"HTTP/2"}
        )

    async with httpx.AsyncClient(
        base_url="https://api.narilabs.com", transport=httpx.MockTransport(handle)
    ) as client:
        monkeypatch.setattr(nari, "get_shared_client", lambda *args: client)
        monkeypatch.setattr(nari, "time", SimpleNamespace(monotonic=lambda: stream.now))
        provider = nari.NariTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)
        result = await provider.synthesize("Hello from Nari.")

    assert result.error is None
    assert (result.provider, result.model, result.voice) == ("nari", _MODEL, _VOICE)
    assert result.ttfa_ms == pytest.approx(25.0)
    assert result.http_version == "HTTP/2"
    assert result.submit_to_headers_ms == pytest.approx(5.0)
    assert result.connection_reused is True
    assert result.audio_path is not None
    try:
        with wave.open(str(result.audio_path), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
            assert wav.readframes(wav.getnframes()) == _PCM
    finally:
        result.audio_path.unlink()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "headers", "chunks", "broken", "expected"),
    [
        (
            503,
            {"Content-Type": "application/json", "x-request-id": "req-503"},
            [b'{"error":"no node"}'],
            False,
            'HTTP 503: {"error":"no node"} (x-request-id=req-503)',
        ),
        (200, _PCM_HEADERS, [], False, "without sending audio"),
        (200, {"Content-Type": "audio/wav"}, [_PCM], False, "Expected Nari audio/pcm"),
        (200, _PCM_HEADERS, [_PCM], True, "stream interrupted"),
    ],
    ids=["http-error", "empty", "wav", "broken"],
)
async def test_failed_or_incompatible_stream_never_saves_audio(
    fake_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    headers: dict[str, str],
    chunks: list[bytes],
    broken: bool,
    expected: str,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=headers, stream=AudioStream(chunks, broken=broken))

    async with httpx.AsyncClient(
        base_url="https://api.narilabs.com", transport=httpx.MockTransport(handle)
    ) as client:
        monkeypatch.setattr(nari, "get_shared_client", lambda *args: client)
        provider = nari.NariTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)
        result = await provider.synthesize("Hello.")

    assert result.error is not None
    assert expected in result.error
    assert result.audio_path is None
    assert result.status_code == status


@pytest.mark.parametrize(
    ("api_key", "model", "match"),
    [
        (None, _MODEL, "nari_api_key"),
        (SecretStr("test-private-key\n"), _MODEL, "nari_api_key"),
        (SecretStr("k"), "qwen3-tts:free", "Invalid Nari TTS model"),
    ],
    ids=["missing-key", "malformed-key", "model"],
)
def test_construction_guards(api_key: SecretStr | None, model: str, match: str) -> None:
    settings = Settings(_env_file=None, nari_api_key=api_key)
    with pytest.raises(ValueError, match=match) as exc:
        nari.NariTTSProvider(settings, model=model, voice=_VOICE)
    assert "test-private-key" not in str(exc.value)


@pytest.mark.asyncio
async def test_warmup_uses_head_without_synthesizing(
    fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(405)

    async with httpx.AsyncClient(
        base_url="https://api.narilabs.com", transport=httpx.MockTransport(handle)
    ) as client:
        monkeypatch.setattr(nari, "get_shared_client", lambda *args: client)
        await nari.NariTTSProvider.warmup(fake_settings)

    assert [(r.method, str(r.url), r.content) for r in requests] == [
        ("HEAD", "https://api.narilabs.com/v1/audio/speech", b"")
    ]


def test_nari_has_a_provider_env_entry() -> None:
    """Without this, publishing Nari breaks the arena key parity check."""
    assert PROVIDER_ENV["nari"] == "NARI_API_KEY"
