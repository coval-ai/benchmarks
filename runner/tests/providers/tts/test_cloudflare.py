# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import wave
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts import cloudflare

from .conftest import make_pcm_bytes

_MODEL = "aura-2-en"
_VOICE = "luna"
_URL = "https://api.cloudflare.com/client/v4/accounts/acct-0123/ai/run/@cf/deepgram/aura-2-en"
_PCM = make_pcm_bytes()


class AudioStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.now = 10.0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            self.now = 10.025 if index == 0 else 12.0
            yield chunk


@pytest.mark.asyncio
async def test_request_shape_and_first_chunk_timing(
    fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = AudioStream([_PCM[:240], _PCM[240:]])

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == _URL
        assert request.headers["Authorization"] == "Bearer test-cloudflare-key"
        assert json.loads(request.content) == {
            "text": "Hello from the edge.",
            "speaker": _VOICE,
            "encoding": "linear16",
            "sample_rate": 24000,
            "container": "none",
        }
        request.extensions.update(__t_submit=10.0, __t_headers=10.005, __connection_reused=True)
        # Workers AI mislabels the PCM body; the provider must not reject it.
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/mpeg"},
            stream=stream,
            extensions={"http_version": b"HTTP/2"},
        )

    async with httpx.AsyncClient(
        base_url="https://api.cloudflare.com", transport=httpx.MockTransport(handle)
    ) as client:
        monkeypatch.setattr(cloudflare, "get_shared_client", lambda *args: client)
        monkeypatch.setattr("time.monotonic", lambda: stream.now)
        provider = cloudflare.CloudflareTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)
        result = await provider.synthesize("Hello from the edge.")

    assert result.error is None
    assert (result.provider, result.model, result.voice) == ("cloudflare", _MODEL, _VOICE)
    assert provider.name == "cloudflare-aura-2-en"
    assert result.ttfa_ms == pytest.approx(25.0)
    assert result.submit_to_headers_ms == pytest.approx(5.0)
    assert result.audio_path is not None
    try:
        with wave.open(str(result.audio_path), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
            assert wav.readframes(wav.getnframes()) == _PCM
    finally:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_workers_ai_error_body_is_surfaced(
    fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {"errors": [{"message": "AiError: Bad input: enum nobody not in luna,thalia"}]}

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, headers={"cf-ai-req-id": "req-400"}, json=body)

    async with httpx.AsyncClient(
        base_url="https://api.cloudflare.com", transport=httpx.MockTransport(handle)
    ) as client:
        monkeypatch.setattr(cloudflare, "get_shared_client", lambda *args: client)
        provider = cloudflare.CloudflareTTSProvider(fake_settings, model=_MODEL, voice="nobody")
        result = await provider.synthesize("Hello.")

    assert result.error == (
        "HTTP 400: AiError: Bad input: enum nobody not in luna,thalia (cf-ai-req-id=req-400)"
    )
    assert result.audio_path is None
    assert result.status_code == 400


@pytest.mark.parametrize(
    ("api_key", "account_id", "match"),
    [
        (None, "acct", "cloudflare_api_key"),
        (SecretStr("k"), None, "cloudflare_account_id"),
    ],
    ids=["missing-key", "missing-account"],
)
def test_construction_guards(api_key: SecretStr | None, account_id: str | None, match: str) -> None:
    settings = Settings(
        _env_file=None, cloudflare_api_key=api_key, cloudflare_account_id=account_id
    )
    with pytest.raises(ValueError, match=match):
        cloudflare.CloudflareTTSProvider(settings, model=_MODEL, voice=_VOICE)
