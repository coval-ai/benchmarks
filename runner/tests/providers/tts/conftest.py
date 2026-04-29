# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for TTS provider tests.

No live network calls are made in any test in this package.  HTTP-streaming
providers are covered by VCR cassettes; WebSocket providers are covered by
monkeypatched fake SDKs.
"""

from __future__ import annotations

import wave as _wave
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from coval_bench.config import Settings

# ---------------------------------------------------------------------------
# VCR config — must be module-scope so pytest-vcr picks it up
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    """Filter secrets from VCR cassettes before they are written to disk."""
    return {
        "filter_headers": ["authorization", "xi-api-key", "api-key", "x-api-key"],
        "filter_query_parameters": ["api_key", "key"],
        "decode_compressed_response": True,
    }


# ---------------------------------------------------------------------------
# Settings fixture with fake API keys
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_settings(tmp_path: Path) -> Settings:
    """Settings instance with placeholder API keys suitable for offline tests."""
    return Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",  # type: ignore[arg-type]
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
        openai_api_key="sk-test-openai",  # type: ignore[arg-type]
        cartesia_api_key="test-cartesia-key",  # type: ignore[arg-type]
        elevenlabs_api_key="test-elevenlabs-key",  # type: ignore[arg-type]
        deepgram_api_key="test-deepgram-key",  # type: ignore[arg-type]
        hume_api_key="test-hume-key",  # type: ignore[arg-type]
        rime_api_key="test-rime-key",  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Sample text fixture
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "Hello, this is a test of the text-to-speech system."


@pytest.fixture()
def sample_text() -> str:
    """Short TTS test sentence."""
    return SAMPLE_TEXT


# ---------------------------------------------------------------------------
# FakeWebSocket — reusable for OpenAI realtime, and ElevenLabs WS tests
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """Minimal async context-manager fake for websockets.connect()."""

    def __init__(self, messages: list[str | bytes]) -> None:
        self._messages = list(messages)
        self._idx = 0
        self.sent: list[str | bytes] = []

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data)

    async def recv(self) -> str | bytes:
        if self._idx < len(self._messages):
            msg = self._messages[self._idx]
            self._idx += 1
            return msg
        raise StopAsyncIteration

    async def __aenter__(self) -> FakeWebSocket:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Fake Cartesia SDK client helpers
# ---------------------------------------------------------------------------


class FakeCartesiaEvent:
    """Mimics a cartesia WebSocket chunk event."""

    def __init__(self, audio: bytes | None = None, event_type: str = "chunk") -> None:
        self.type = event_type
        self.audio = audio


class FakeCartesiaContext:
    """Fake AsyncWebSocketContext that yields audio chunks then done."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def send(self, **_kwargs: Any) -> None:
        pass

    async def no_more_inputs(self) -> None:
        pass

    def receive(self) -> FakeCartesiaContext:
        return self

    def __aiter__(self) -> FakeCartesiaContext:
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> FakeCartesiaEvent:
        try:
            chunk = next(self._iter)
            return FakeCartesiaEvent(audio=chunk, event_type="chunk")
        except StopIteration:
            raise StopAsyncIteration from None


class FakeCartesiaConnection:
    """Fake AsyncTTSResourceConnection."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def context(self, **_kwargs: Any) -> FakeCartesiaContext:
        return FakeCartesiaContext(self._chunks)

    async def __aenter__(self) -> FakeCartesiaConnection:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


def make_fake_cartesia_client(chunks: list[bytes]) -> MagicMock:
    """Return a MagicMock AsyncCartesia whose websocket_connect yields *chunks*."""
    fake_conn = FakeCartesiaConnection(chunks)
    fake_ws_manager = MagicMock()
    fake_ws_manager.__aenter__ = AsyncMock(return_value=fake_conn)
    fake_ws_manager.__aexit__ = AsyncMock(return_value=False)

    fake_tts = MagicMock()
    fake_tts.websocket_connect = MagicMock(return_value=fake_ws_manager)

    fake_client = MagicMock()
    fake_client.tts = fake_tts
    return fake_client


# ---------------------------------------------------------------------------
# Fake ElevenLabs SDK helpers
# ---------------------------------------------------------------------------


def make_fake_elevenlabs_response(chunks: list[bytes]) -> list[bytes]:
    """Return a list of byte chunks to be yielded by the SDK iterator."""
    return list(chunks)


# ---------------------------------------------------------------------------
# Fake aiohttp response helpers
# ---------------------------------------------------------------------------


class FakeAiohttpContent:
    """Fake aiohttp response content that yields *chunks* from iter_any()."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self) -> Any:  # noqa: ANN401
        for chunk in self._chunks:
            yield chunk


class FakeAiohttpResponse:
    """Fake aiohttp.ClientResponse."""

    def __init__(self, chunks: list[bytes], status: int = 200, text_body: str = "") -> None:
        self.status = status
        self.content = FakeAiohttpContent(chunks)
        self._text_body = text_body

    async def text(self) -> str:
        return self._text_body

    async def __aenter__(self) -> FakeAiohttpResponse:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


class FakeAiohttpSession:
    """Fake aiohttp.ClientSession that returns *response* for any post()."""

    def __init__(self, response: FakeAiohttpResponse) -> None:
        self._response = response

    def post(self, *_args: Any, **_kwargs: Any) -> FakeAiohttpResponse:
        return self._response

    async def __aenter__(self) -> FakeAiohttpSession:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers — build minimal valid WAV bytes for fixture audio
# ---------------------------------------------------------------------------


def make_pcm_bytes(duration_frames: int = 480, sample_rate: int = 24000) -> bytes:
    """Generate a block of silence PCM frames (16-bit, mono)."""
    return b"\x00" * (duration_frames * 2)  # 2 bytes per sample, 16-bit


def make_wav_bytes(duration_frames: int = 480, sample_rate: int = 24000) -> bytes:
    """Wrap PCM silence in a proper WAV container."""
    buf = BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(make_pcm_bytes(duration_frames, sample_rate))
    return buf.getvalue()
