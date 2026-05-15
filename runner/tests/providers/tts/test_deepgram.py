# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Deepgram TTS provider — WebSocket Speak API via ``ws_client.connect``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.deepgram import SAMPLE_RATE, DeepgramTTSProvider
from tests.providers.stt.conftest import FakeWebSocket
from tests.providers.tts.conftest import make_pcm_bytes


def _fake_connect(ws: FakeWebSocket) -> Any:
    """Return an async context manager that yields *ws* (matches ``websockets.connect``)."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _speak_fixture_events(pcm_chunks: list[bytes]) -> list[Any]:
    """Event stream: Metadata (first ``recv``) then PCM + Flushed for ``async for ws``."""
    return [{"type": "Metadata"}, *pcm_chunks, {"type": "Flushed"}]


@pytest.mark.asyncio
async def test_deepgram_happy_path(fake_settings: Settings) -> None:
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_speak_fixture_events(chunks))
    provider = DeepgramTTSProvider(
        fake_settings, model="aura-2-thalia-en", voice="aura-2-thalia-en"
    )

    with patch(
        "coval_bench.providers.tts.deepgram.ws_client.connect",
        return_value=_fake_connect(ws),
    ):
        result = await provider.synthesize("Hello from Deepgram")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "deepgram"
    assert result.model == "aura-2-thalia-en"

    sent_strs = [s.decode() if isinstance(s, bytes) else s for s in ws._sent]
    assert any('"type": "Speak"' in s for s in sent_strs)
    assert any('"type": "Flush"' in s for s in sent_strs)

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_deepgram_connect_refused(fake_settings: Settings) -> None:
    """WebSocket connect failure → result.error populated, audio_path None."""

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("connection refused")

    provider = DeepgramTTSProvider(
        fake_settings, model="aura-2-thalia-en", voice="aura-2-thalia-en"
    )

    with patch("coval_bench.providers.tts.deepgram.ws_client.connect", side_effect=_boom):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "refused" in result.error.lower()
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_deepgram_network_error(fake_settings: Settings) -> None:
    """Unexpected exception during the WS session → result.error set."""

    provider = DeepgramTTSProvider(
        fake_settings, model="aura-2-thalia-en", voice="aura-2-thalia-en"
    )

    class _BadWs:
        async def __aenter__(self) -> _BadWs:
            return self

        async def __aexit__(self, *_: object) -> None:
            pass

        async def recv(self) -> str:
            raise RuntimeError("recv failed")

        async def send(self, _data: str | bytes) -> None:
            pass

        def __aiter__(self) -> _BadWs:
            return self

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=_BadWs())
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("coval_bench.providers.tts.deepgram.ws_client.connect", return_value=cm):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_deepgram_no_audio_chunks(fake_settings: Settings) -> None:
    """Metadata + Flushed only (no binary PCM) → audio_path None, ttfa_ms None."""
    ws = FakeWebSocket(_speak_fixture_events([]))
    provider = DeepgramTTSProvider(
        fake_settings, model="aura-2-thalia-en", voice="aura-2-thalia-en"
    )

    with patch(
        "coval_bench.providers.tts.deepgram.ws_client.connect",
        return_value=_fake_connect(ws),
    ):
        result = await provider.synthesize("silent test")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


def test_deepgram_name_and_model(fake_settings: Settings) -> None:
    p = DeepgramTTSProvider(fake_settings, model="aura-2-thalia-en", voice="v")
    assert p.name == "deepgram-aura-2-thalia-en"
    assert p.model == "aura-2-thalia-en"


@pytest.mark.asyncio
async def test_deepgram_warning_then_pcm(fake_settings: Settings) -> None:
    """First ``recv()`` may be a Warning JSON frame; handler logs and continues."""

    pcm = make_pcm_bytes(120)
    # Single recv consumes Warning; async iter gets pcm + Flushed from shared queue.
    events: list[Any] = [
        {"type": "Warning", "description": "test warning", "code": "WARN_TEST"},
        pcm,
        {"type": "Flushed"},
    ]
    ws = FakeWebSocket(events)
    provider = DeepgramTTSProvider(
        fake_settings, model="aura-2-thalia-en", voice="aura-2-thalia-en"
    )

    with patch(
        "coval_bench.providers.tts.deepgram.ws_client.connect",
        return_value=_fake_connect(ws),
    ):
        result = await provider.synthesize("Hi")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    assert result.audio_path.stat().st_size > 0
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_synthesize_wav_contains_riff(fake_settings: Settings) -> None:
    chunks = [
        make_pcm_bytes(240),
        make_pcm_bytes(240),
        make_pcm_bytes(240),
    ]
    ws = FakeWebSocket(_speak_fixture_events(chunks))
    provider = DeepgramTTSProvider(
        fake_settings,
        model="aura-2-thalia-en",
        voice="aura-2-thalia-en",
    )

    captured: dict[str, str] = {}

    def _capture(url: str, **_kwargs: object) -> Any:
        captured["url"] = url
        return _fake_connect(ws)

    with patch("coval_bench.providers.tts.deepgram.ws_client.connect", side_effect=_capture):
        result = await provider.synthesize("Hello world")

    assert urlparse(captured["url"]).hostname == "api.deepgram.com"
    assert f"sample_rate={SAMPLE_RATE}" in captured["url"]
    assert "encoding=linear16" in captured["url"]

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.provider == "deepgram"
    assert result.model == "aura-2-thalia-en"
    assert result.voice == "aura-2-thalia-en"
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    result.audio_path.unlink()


def test_deepgram_missing_api_key() -> None:
    settings_no_key = Settings(
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        deepgram_api_key=None,
    )
    with pytest.raises(ValueError, match="deepgram_api_key"):
        DeepgramTTSProvider(settings_no_key, model="aura-2-thalia-en", voice="v")
