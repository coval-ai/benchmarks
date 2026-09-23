# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
import wave
from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts import gemini

from .conftest import make_pcm_bytes

_MODEL = "gemini-3.8-flash-tts"
_VOICE = "Kore"
_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
_PCM = make_pcm_bytes()


def _sse(event_type: str, **fields: object) -> bytes:
    data = json.dumps({**fields, "event_type": event_type})
    return f"event: {event_type}\ndata: {data}\n\n".encode()


def _audio_delta(pcm: bytes, **overrides: object) -> bytes:
    delta = {
        "type": "audio",
        "mime_type": "audio/l16",
        "sample_rate": 24000,
        "channels": 1,
        "data": base64.b64encode(pcm).decode(),
        **overrides,
    }
    return _sse("step.delta", index=0, delta=delta)


_PREAMBLE = _sse("interaction.created", interaction={"id": "v1_x"}) + _sse("step.start", index=0)
_EPILOGUE = (
    _sse("step.stop", index=0)
    + _sse("interaction.completed", interaction={"id": "v1_x"})
    + b"event: done\ndata: [DONE]\n\n"
)


class SseStream(httpx.AsyncByteStream):
    def __init__(self, frames: list[bytes], *, broken: bool = False) -> None:
        self.frames = frames
        self.broken = broken
        self.now = 10.0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, frame in enumerate(self.frames):
            self.now = 10.025 if index == 0 else 12.0
            yield frame
        if self.broken:
            raise httpx.ReadError("stream interrupted")


@pytest.mark.asyncio
async def test_request_shape_and_first_chunk_timing(
    fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = SseStream([_PREAMBLE + _audio_delta(_PCM[:240]), _audio_delta(_PCM[240:]) + _EPILOGUE])

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == _URL
        assert request.headers["x-goog-api-key"] == "test-gemini-key"
        assert json.loads(request.content) == {
            "model": _MODEL,
            "input": "Hello from Gemini.",
            "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
            "generation_config": {"speech_config": [{"voice": _VOICE}]},
            "stream": True,
        }
        request.extensions.update(__t_submit=10.0, __t_headers=10.005, __connection_reused=True)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=stream,
            extensions={"http_version": b"HTTP/2"},
        )

    async with httpx.AsyncClient(
        base_url="https://generativelanguage.googleapis.com",
        transport=httpx.MockTransport(handle),
    ) as client:
        monkeypatch.setattr(gemini, "get_shared_client", lambda *args: client)
        monkeypatch.setattr(gemini, "time", SimpleNamespace(monotonic=lambda: stream.now))
        provider = gemini.GeminiTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)
        result = await provider.synthesize("Hello from Gemini.")

    assert result.error is None
    assert (result.provider, result.model, result.voice) == ("gemini", _MODEL, _VOICE)
    assert provider.name == "gemini-gemini-3.8-flash-tts"
    assert result.ttfa_ms == pytest.approx(25.0)
    assert result.submit_to_headers_ms == pytest.approx(5.0)
    assert result.connection_reused is True
    assert result.audio_path is not None
    try:
        with wave.open(str(result.audio_path), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
            assert wav.readframes(wav.getnframes()) == _PCM
    finally:
        result.audio_path.unlink()


_BAD_KEY_BODY = json.dumps([{"error": {"code": 400, "message": "API key not valid."}}])
_NOT_FOUND_BODY = json.dumps({"error": {"message": "Model 'x' not found.", "code": "not_found"}})
_VOICE_ERROR = _sse(
    "error", error={"message": "No matching speaker voice found", "code": "invalid_request"}
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "frames", "broken", "expected"),
    [
        (400, [_BAD_KEY_BODY.encode()], False, "HTTP 400: API key not valid."),
        (404, [_NOT_FOUND_BODY.encode()], False, "HTTP 404: Model 'x' not found."),
        (200, [_PREAMBLE + _VOICE_ERROR], False, "No matching speaker voice found"),
        (200, [_PREAMBLE + _EPILOGUE], False, None),
        (200, [_PREAMBLE + _audio_delta(_PCM, sample_rate=16000) + _EPILOGUE], False, None),
        (200, [_PREAMBLE + _audio_delta(_PCM)], True, "stream interrupted"),
    ],
    ids=["bad-key", "not-found", "voice-error-event", "empty", "sample-rate", "broken"],
)
async def test_failed_or_incompatible_stream_never_saves_audio(
    fake_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    frames: list[bytes],
    broken: bool,
    expected: str | None,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, stream=SseStream(frames, broken=broken))

    async with httpx.AsyncClient(
        base_url="https://generativelanguage.googleapis.com",
        transport=httpx.MockTransport(handle),
    ) as client:
        monkeypatch.setattr(gemini, "get_shared_client", lambda *args: client)
        provider = gemini.GeminiTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)
        result = await provider.synthesize("Hello.")

    assert result.error
    if expected is not None:
        assert result.error == expected
    assert result.audio_path is None
    assert result.status_code == status


@pytest.mark.parametrize("api_key", [None, SecretStr("")])
def test_missing_api_key_is_rejected(api_key: SecretStr | None) -> None:
    settings = Settings(_env_file=None, gemini_api_key=api_key)
    with pytest.raises(ValueError, match="gemini_api_key"):
        gemini.GeminiTTSProvider(settings, model=_MODEL, voice=_VOICE)


@pytest.mark.asyncio
async def test_warmup_uses_head_without_synthesizing(
    fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(404)

    async with httpx.AsyncClient(
        base_url="https://generativelanguage.googleapis.com",
        transport=httpx.MockTransport(handle),
    ) as client:
        monkeypatch.setattr(gemini, "get_shared_client", lambda *args: client)
        await gemini.GeminiTTSProvider.warmup(fake_settings)

    assert [(r.method, str(r.url), r.content) for r in requests] == [("HEAD", _URL, b"")]
