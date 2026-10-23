# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for STT provider tests.

Audio fixture note
------------------
``fixtures/audio/sample-16k-mono.wav`` is a deterministic 3-second 440 Hz sine
wave at 16 kHz mono PCM_16 (~96 KB).  Regenerate with::

    python -c "
    import numpy as np, soundfile as sf
    t = np.linspace(0, 3.0, 48000, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
    sf.write(
        'tests/providers/stt/fixtures/audio/sample-16k-mono.wav',
        audio, 16000, subtype='PCM_16',
    )
    "

``FakeWebSocket`` pattern
-------------------------
Each WS-based provider test patches ``websockets.asyncio.client.connect`` with a
``FakeWebSocket`` that replays JSON events loaded from
``fixtures/<provider>/events-*.json``.  No real sockets are opened.
``pytest-socket`` is configured in ``pyproject.toml`` dev deps and should be
used with ``--disable-socket --allow-unix-socket`` in CI to enforce this.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.stt import STT_PROVIDERS
from coval_bench.providers.stt._streaming import WebSocketSTTProvider

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_FIXTURE = FIXTURES_DIR / "audio" / "sample-16k-mono.wav"


# ---------------------------------------------------------------------------
# FakeWebSocket
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """Async context-manager WebSocket fake that replays JSON events from a list.

    Each entry in *events* is either a ``dict`` (serialised to JSON) or a
    ``str`` (sent verbatim).  ``bytes`` entries are forwarded as-is. With
    ``server_closes=False`` the receive iterator blocks until the client closes,
    modelling a server (e.g. Inworld) that keeps the socket open.
    """

    def __init__(
        self,
        events: list[dict[str, Any] | str | bytes],
        on_send: Callable[[Any], None] | None = None,
        *,
        server_closes: bool = True,
    ) -> None:
        self._events: list[dict[str, Any] | str | bytes] = list(events)
        self._on_send = on_send
        self._server_closes = server_closes
        self._closed = asyncio.Event()
        self._sent: list[Any] = []

    async def __aenter__(self) -> FakeWebSocket:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._closed.set()

    async def close(self) -> None:
        self._closed.set()
        self._events.clear()

    async def send(self, msg: bytes | str) -> None:
        if self._closed.is_set():
            raise ConnectionError("send on a closed FakeWebSocket")
        self._sent.append(msg)
        if self._on_send is not None:
            self._on_send(msg)

    async def recv(self) -> str | bytes:
        if not self._events:
            raise StopAsyncIteration
        evt = self._events.pop(0)
        if isinstance(evt, dict):
            return json.dumps(evt)
        return evt

    def __aiter__(self) -> AsyncFakeWebSocketIter:
        return AsyncFakeWebSocketIter(self._events, self._closed, self._server_closes)


class AsyncFakeWebSocketIter:
    """Async iterator over the remaining events list."""

    def __init__(
        self,
        events: list[dict[str, Any] | str | bytes],
        closed: asyncio.Event,
        server_closes: bool,
    ) -> None:
        self._events = events
        self._closed = closed
        self._server_closes = server_closes

    def __aiter__(self) -> AsyncFakeWebSocketIter:
        return self

    async def __anext__(self) -> str | bytes:
        while not self._events:
            if self._server_closes or self._closed.is_set():
                raise StopAsyncIteration
            await self._closed.wait()
        evt = self._events.pop(0)
        if isinstance(evt, dict):
            return json.dumps(evt)
        if isinstance(evt, (str, bytes)):
            return evt
        raise StopAsyncIteration


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

_REAL_SLEEP = asyncio.sleep


@pytest.fixture(autouse=True)
def _fast_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero out pacing sleeps and final-event waits; sleeps still yield to the loop."""

    async def _instant(_delay: float, result: Any = None) -> Any:
        return await _REAL_SLEEP(0, result)

    monkeypatch.setattr(asyncio, "sleep", _instant)
    # The fake never answers an end-of-audio frame, so every post-audio wait times out.
    for cls in (WebSocketSTTProvider, *STT_PROVIDERS.values()):
        if "_FINAL_WAIT_S" in vars(cls):
            monkeypatch.setattr(cls, "_FINAL_WAIT_S", 0.05)


_TEST_KEY = SecretStr("test-api-key-00000000000000000000")

_STT_KEY_FIELDS = (
    "assemblyai_api_key",
    "azure_api_key",
    "baseten_api_key",
    "cartesia_api_key",
    "cloudflare_api_key",
    "deepgram_api_key",
    "elevenlabs_api_key",
    "gemini_api_key",
    "gladia_api_key",
    "gradium_api_key",
    "guava_api_key",
    "inworld_api_key",
    "meta_api_key",
    "mistral_api_key",
    "modulate_api_key",
    "nari_api_key",
    "openai_api_key",
    "reson8_api_key",
    "revai_api_key",
    "smallest_api_key",
    "soniox_api_key",
    "speechmatics_api_key",
    "stepfun_api_key",
    "together_api_key",
    "xai_api_key",
    "zoom_api_key",
    "zoom_api_secret",
)


def make_settings(**overrides: Any) -> Settings:
    """Settings with every STT credential set and the local .env ignored."""
    values: dict[str, Any] = {field: _TEST_KEY for field in _STT_KEY_FIELDS}
    values.update(
        azure_region="eastus",
        cloudflare_account_id="acct-test",
        google_project_id="proj-test",
        guava_base_url="https://guava.test",
        baseten_whisper_url="wss://whisper.test/v1/websocket",
        baseten_qwen_asr_url="wss://qwen.test/v1/websocket",
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def stt_settings() -> Settings:
    return make_settings()


@pytest.fixture
def fake_api_key() -> SecretStr:
    return _TEST_KEY


def fake_connect(
    events: list[Any],
    sent: list[Any] | None = None,
    *,
    server_closes: bool = True,
) -> Any:
    """A ``ws_client.connect`` stand-in: an async CM yielding a FakeWebSocket."""
    from unittest.mock import AsyncMock, MagicMock

    ws = FakeWebSocket(
        events, on_send=(sent.append if sent is not None else None), server_closes=server_closes
    )
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    cm.ws = ws
    return cm


@pytest.fixture
def audio_pcm_bytes() -> bytes:
    """Raw PCM bytes from the 3-second 440 Hz fixture WAV."""
    import soundfile as sf

    data, _sr = sf.read(str(AUDIO_FIXTURE), dtype="int16", always_2d=False)
    return bytes(data.tobytes())


def load_fixture_events(provider: str, scenario: str = "events-success") -> list[Any]:
    """Load JSON event list from ``fixtures/<provider>/<scenario>.json``."""
    path = FIXTURES_DIR / provider / f"{scenario}.json"
    return cast(list[Any], json.loads(path.read_text()))
