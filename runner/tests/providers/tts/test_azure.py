# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Azure AI Speech WebSocket TTS provider."""

from __future__ import annotations

import json
import struct
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.azure import AzureTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_VOICE = "en-US-AvaNeural"


def _text_frame(path: str, body: str = "{}") -> str:
    return f"Path:{path}\r\nX-RequestId:req\r\nContent-Type:application/json\r\n\r\n{body}"


def _binary_frame(payload: bytes, path: str = "audio") -> bytes:
    header = f"Path:{path}\r\nX-RequestId:req\r\nContent-Type:audio/x-wav\r\n".encode("ascii")
    return struct.pack(">H", len(header)) + header + payload


def _synthesis_frames(pcm_chunks: list[bytes]) -> list[str | bytes]:
    frames: list[str | bytes] = [_text_frame("turn.start")]
    frames.extend(_binary_frame(chunk) for chunk in pcm_chunks)
    frames.append(_text_frame("turn.end"))
    return frames


def _sent_text_frames(ws: FakeWebSocket) -> list[tuple[dict[str, str], str]]:
    """Parse client text frames into (headers, body) pairs."""
    parsed: list[tuple[dict[str, str], str]] = []
    for message in ws.sent:
        assert isinstance(message, str)
        head, _, body = message.partition("\r\n\r\n")
        headers: dict[str, str] = {}
        for line in head.split("\r\n"):
            name, _, value = line.partition(":")
            headers[name.strip()] = value.strip()
        parsed.append((headers, body))
    return parsed


class AbortingWebSocket(FakeWebSocket):
    """Yields its frames, then fails like an abnormally closed connection."""

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        async def gen() -> AsyncIterator[str | bytes]:
            for message in self._messages:
                yield message
            raise RuntimeError("connection closed: internal server error")

        return gen()


@pytest.mark.asyncio
async def test_azure_tts_happy_path(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_synthesis_frames([make_pcm_bytes(240), make_pcm_bytes(240)]))
    provider = AzureTTSProvider(fake_settings, model="neural", voice=_VOICE)

    with patch("coval_bench.providers.tts.azure.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Azure")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.provider == "azure"
    assert result.model == "neural"
    assert result.voice == _VOICE
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_azure_tts_url_and_subscription_key_header(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_synthesis_frames([make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return ws

    provider = AzureTTSProvider(fake_settings, model="neural", voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.azure.ws_client.connect", side_effect=connect_side_effect
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert captured["url"] == "wss://eastus.tts.speech.microsoft.com/cognitiveservices/websocket/v1"
    headers = captured["kwargs"]["additional_headers"]  # type: ignore[index]
    assert headers["Ocp-Apim-Subscription-Key"] == "test-azure-key"
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_azure_tts_sends_config_context_then_ssml(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_synthesis_frames([make_pcm_bytes(240)]))
    provider = AzureTTSProvider(fake_settings, model="neural", voice=_VOICE)

    with patch("coval_bench.providers.tts.azure.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello world")

    frames = _sent_text_frames(ws)
    assert [headers["Path"] for headers, _ in frames] == [
        "speech.config",
        "synthesis.context",
        "ssml",
    ]
    request_ids = {headers["X-RequestId"] for headers, _ in frames}
    assert len(request_ids) == 1

    context = json.loads(frames[1][1])
    assert context["synthesis"]["audio"]["outputFormat"] == "raw-24khz-16bit-mono-pcm"

    ssml_headers, ssml_body = frames[2]
    assert ssml_headers["Content-Type"] == "application/ssml+xml"
    assert f'name="{_VOICE}"' in ssml_body
    assert ">Hello world<" in ssml_body
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_azure_tts_ssml_escapes_text(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_synthesis_frames([make_pcm_bytes(240)]))
    provider = AzureTTSProvider(fake_settings, model="neural", voice=_VOICE)

    with patch("coval_bench.providers.tts.azure.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Tom & Jerry <3")

    _, ssml_body = _sent_text_frames(ws)[2]
    assert "Tom &amp; Jerry &lt;3" in ssml_body
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_azure_tts_hd_voice_name_in_ssml(fake_settings: Settings) -> None:
    hd_voice = "en-US-Ava:DragonHDLatestNeural"
    ws = FakeWebSocket(_synthesis_frames([make_pcm_bytes(240)]))
    provider = AzureTTSProvider(fake_settings, model="dragon-hd-latest", voice=hd_voice)

    with patch("coval_bench.providers.tts.azure.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    _, ssml_body = _sent_text_frames(ws)[2]
    assert f'name="{hd_voice}"' in ssml_body
    assert result.voice == hd_voice
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_azure_tts_ttfa_set_on_first_chunk_only(fake_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_synthesis_frames(chunks))
    provider = AzureTTSProvider(fake_settings, model="neural", voice=_VOICE)

    times = iter([0.0, 0.1])

    with (
        patch(
            "coval_bench.providers.tts.azure.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch("coval_bench.providers.tts.azure.ws_client.connect", return_value=ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_azure_tts_skips_empty_audio_frames(fake_settings: Settings) -> None:
    frames: list[str | bytes] = [
        _text_frame("turn.start"),
        _binary_frame(b""),  # header-only audio frame marks end of audio stream
        _text_frame("turn.end"),
    ]
    ws = FakeWebSocket(frames)
    provider = AzureTTSProvider(fake_settings, model="neural", voice=_VOICE)

    with patch("coval_bench.providers.tts.azure.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_azure_tts_ignores_metadata_frames(fake_settings: Settings) -> None:
    frames: list[str | bytes] = [
        _text_frame("turn.start"),
        _text_frame("response", json.dumps({"audio": {"type": "inline"}})),
        _text_frame("audio.metadata", json.dumps({"Metadata": []})),
        _binary_frame(make_pcm_bytes(240)),
        _text_frame("turn.end"),
    ]
    ws = FakeWebSocket(frames)
    provider = AzureTTSProvider(fake_settings, model="neural", voice=_VOICE)

    with patch("coval_bench.providers.tts.azure.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_azure_tts_abnormal_close_after_partial_audio(fake_settings: Settings) -> None:
    ws = AbortingWebSocket([_text_frame("turn.start"), _binary_frame(make_pcm_bytes(240))])
    provider = AzureTTSProvider(fake_settings, model="neural", voice=_VOICE)

    with patch("coval_bench.providers.tts.azure.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is not None
    assert "internal server error" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is not None


@pytest.mark.asyncio
async def test_azure_tts_stream_end_without_turn_end(fake_settings: Settings) -> None:
    ws = FakeWebSocket([_text_frame("turn.start"), _binary_frame(make_pcm_bytes(240))])
    provider = AzureTTSProvider(fake_settings, model="neural", voice=_VOICE)

    with patch("coval_bench.providers.tts.azure.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_azure_tts_invalid_model_raises(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Azure TTS model"):
        AzureTTSProvider(fake_settings, model="not-a-model", voice=_VOICE)


def test_azure_tts_empty_voice_raises(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="requires a voice name"):
        AzureTTSProvider(fake_settings, model="neural", voice="")


def test_azure_tts_missing_api_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        azure_api_key=None,
        azure_region="eastus",
    )
    with pytest.raises(ValueError, match="azure_api_key is required"):
        AzureTTSProvider(settings, model="neural", voice=_VOICE)


def test_azure_tts_missing_region_raises(fake_settings: Settings) -> None:
    settings = fake_settings.model_copy(update={"azure_region": None})
    with pytest.raises(ValueError, match="azure_region is required"):
        AzureTTSProvider(settings, model="neural", voice=_VOICE)


def test_azure_tts_provider_name(fake_settings: Settings) -> None:
    provider = AzureTTSProvider(fake_settings, model="neural", voice=_VOICE)
    assert provider.name == "azure-neural"
    assert provider.model == "neural"
