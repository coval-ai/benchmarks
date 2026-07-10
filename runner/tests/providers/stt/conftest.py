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

# Final/flush waits the FakeWebSocket never releases.
_WAIT_PATCHES = (
    "coval_bench.providers.stt.assemblyai._FINAL_WAIT_S",
    "coval_bench.providers.stt.deepgram._FINAL_WAIT_S",
    "coval_bench.providers.stt.deepgram._FLUX_EOT_SILENCE_S",
    "coval_bench.providers.stt.revai._EOT_SILENCE_S",
    "coval_bench.providers.stt.xai._FINAL_WAIT_S",
    "coval_bench.providers.stt.gradium._FLUSH_WAIT_S",
    "coval_bench.providers.stt.inworld._CLOSE_WAIT_S",
    "coval_bench.providers.stt.together._FINAL_WAIT_S",
    "coval_bench.providers.stt.together._TAIL_SILENCE_S",
)


@pytest.fixture(autouse=True)
def _fast_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero out pacing sleeps and final-event waits; sleeps still yield to the loop."""

    async def _instant(_delay: float, result: Any = None) -> Any:
        return await _REAL_SLEEP(0, result)

    monkeypatch.setattr(asyncio, "sleep", _instant)
    for target in _WAIT_PATCHES:
        monkeypatch.setattr(target, 0.05)


@pytest.fixture
def fake_api_key() -> SecretStr:
    return SecretStr("test-api-key-00000000000000000000")


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
