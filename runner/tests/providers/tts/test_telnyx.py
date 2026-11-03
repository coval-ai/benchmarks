# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Telnyx TTS provider (chunked HTTP streaming)."""

from __future__ import annotations

import base64
import json
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx2
import pytest

from coval_bench.config import Settings
from coval_bench.providers import _http_session
from coval_bench.providers.tts import TTS_PROVIDERS
from coval_bench.providers.tts.telnyx import TelnyxTTSProvider
from tests.providers.fakes import fake_connect

from .conftest import make_pcm_bytes

_MODEL = "Telnyx.Ultra"
_VOICE = "Clara"


def _install_mock(handler: Any) -> None:
    _http_session._CLIENTS["telnyx"] = httpx2.AsyncClient(
        base_url="https://api.telnyx.com",
        transport=httpx2.MockTransport(handler),
    )


@pytest.fixture(autouse=True)
def reset_clients() -> Generator[None, None, None]:
    _http_session._CLIENTS.clear()
    yield
    _http_session._CLIENTS.clear()


def test_telnyx_is_registered() -> None:
    assert TTS_PROVIDERS["telnyx"] is TelnyxTTSProvider


@pytest.mark.asyncio
async def test_telnyx_tts_happy_path(fake_settings: Settings) -> None:
    pcm = make_pcm_bytes(480) * 4
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.read())
        return httpx2.Response(200, content=pcm, headers={"content-type": "audio/pcm"})

    _install_mock(handler)
    provider = TelnyxTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)
    result = await provider.synthesize("Hello from Telnyx")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 60_000
    assert result.provider == "telnyx"
    assert result.model == _MODEL
    assert result.voice == _VOICE
    assert result.audio_path is not None
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    result.audio_path.unlink()

    assert captured["url"] == "https://api.telnyx.com/v2/text-to-speech/speech"
    assert captured["authorization"] == "Bearer test-telnyx-key"
    assert captured["body"] == {
        "text": "Hello from Telnyx",
        "voice": "Telnyx.Ultra.Clara",
        "output_type": "binary_output",
        "disable_cache": True,
        "telnyx": {"response_format": "pcm", "sampling_rate": 24000},
    }


@pytest.mark.asyncio
async def test_telnyx_tts_third_party_vendor_params(fake_settings: Settings) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.read())
        return httpx2.Response(200, content=make_pcm_bytes(480))

    _install_mock(handler)
    provider = TelnyxTTSProvider(fake_settings, model="xAI", voice="eve")
    result = await provider.synthesize("Hi")

    assert result.error is None, result.error
    assert captured["body"]["voice"] == "xAI.eve"
    assert captured["body"]["xai"] == {"output_format": "pcm", "sample_rate": 24000}
    if result.audio_path:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_telnyx_tts_http_error(fake_settings: Settings) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"errors": [{"code": "10015", "detail": "unknown voice"}]})

    _install_mock(handler)
    provider = TelnyxTTSProvider(fake_settings, model=_MODEL, voice="nobody")
    result = await provider.synthesize("Hello")

    assert result.error is not None
    assert result.error.startswith("HTTP 400")
    assert "unknown voice" in result.error
    assert result.status_code == 400
    assert result.audio_path is None
    assert result.ttfa_ms is None


def test_telnyx_tts_rejects_unmapped_vendor(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="no PCM output mapping"):
        TelnyxTTSProvider(fake_settings, model="Minimax.speech-2.8-turbo", voice="x")


def test_telnyx_tts_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="telnyx_api_key is required"):
        TelnyxTTSProvider(Settings(telnyx_api_key=None), model=_MODEL, voice=_VOICE)


@pytest.mark.asyncio
async def test_telnyx_tts_warmup_heads_speech_endpoint(fake_settings: Settings) -> None:
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx2.Response(401, content=b"unauthorized")

    _install_mock(handler)
    await TelnyxTTSProvider.warmup(fake_settings)
    assert seen == ["HEAD /v2/text-to-speech/speech"]


def _ws_audio(pcm: bytes, text: str | None = None) -> dict[str, Any]:
    return {"audio": base64.b64encode(pcm).decode("ascii"), "text": text, "isFinal": False}


_WS_FINAL: dict[str, Any] = {"audio": None, "text": "", "isFinal": True}


async def _run_ws(fake_settings: Settings, events: list[Any], voice: str = "Delta") -> Any:
    cm = fake_connect(events)
    connect = MagicMock(return_value=cm)
    with patch("coval_bench.providers.tts.telnyx.ws_client.connect", connect):
        provider = TelnyxTTSProvider(fake_settings, model="Telnyx.Qwen3TTS", voice=voice)
        result = await provider.synthesize("Hello from Qwen")
    return result, cm.ws, connect


@pytest.mark.asyncio
async def test_telnyx_tts_qwen_streams_over_websocket(fake_settings: Settings) -> None:
    events: list[Any] = [
        _ws_audio(make_pcm_bytes(480)),
        _ws_audio(make_pcm_bytes(480)),
        {"audio": None, "text": "Hello from Qwen", "isFinal": False},
        _WS_FINAL,
    ]
    result, ws, connect = await _run_ws(fake_settings, events)

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    result.audio_path.unlink()

    url = connect.call_args.args[0]
    assert url.startswith(
        "wss://api.telnyx.com/v2/text-to-speech/speech?voice=Telnyx.Qwen3TTS.Delta"
    )
    assert "audio_format=pcm" in url and "disable_cache=true" in url
    assert connect.call_args.kwargs["additional_headers"] == {
        "Authorization": "Bearer test-telnyx-key"
    }
    assert [json.loads(m) for m in ws.sent] == [
        {"text": " "},
        {"text": "Hello from Qwen"},
        {"text": ""},
    ]


@pytest.mark.asyncio
async def test_telnyx_tts_websocket_error_frame(fake_settings: Settings) -> None:
    result, _, _ = await _run_ws(fake_settings, [{"error": "voice not found"}])

    assert result.error == "voice not found"
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_telnyx_tts_websocket_rejects_mp3(fake_settings: Settings) -> None:
    mp3 = b"ID3\x04" + bytes(64)
    result, _, _ = await _run_ws(fake_settings, [_ws_audio(mp3), _WS_FINAL])

    assert result.error is not None
    assert "MP3" in result.error


@pytest.mark.asyncio
async def test_telnyx_tts_http_rejects_mp3(fake_settings: Settings) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, content=b"ID3\x04" + bytes(64), headers={"content-type": "audio/mpeg"}
        )

    _install_mock(handler)
    provider = TelnyxTTSProvider(fake_settings, model="Telnyx.KokoroTTS", voice="af_heart")
    result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "MP3" in result.error
    assert result.audio_path is None
