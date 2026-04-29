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

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
    ``str`` (sent verbatim).  ``bytes`` entries are forwarded as-is.
    """

    def __init__(
        self,
        events: list[dict[str, Any] | str | bytes],
        on_send: Callable[[Any], None] | None = None,
    ) -> None:
        self._events: list[dict[str, Any] | str | bytes] = list(events)
        self._on_send = on_send
        self._closed = False
        self._sent: list[Any] = []

    async def __aenter__(self) -> FakeWebSocket:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._closed = True

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
        return AsyncFakeWebSocketIter(self._events)


class AsyncFakeWebSocketIter:
    """Async iterator over the remaining events list."""

    def __init__(self, events: list[dict[str, Any] | str | bytes]) -> None:
        self._events = events

    def __aiter__(self) -> AsyncFakeWebSocketIter:
        return self

    async def __anext__(self) -> str | bytes:
        if not self._events:
            raise StopAsyncIteration
        evt = self._events.pop(0)
        if isinstance(evt, dict):
            return json.dumps(evt)
        if isinstance(evt, (str, bytes)):
            return evt
        raise StopAsyncIteration


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_api_key() -> SecretStr:
    return SecretStr("test-api-key-00000000000000000000")


@pytest.fixture
def audio_pcm_bytes() -> bytes:
    """Raw PCM bytes from the 3-second 440 Hz fixture WAV."""
    import soundfile as sf

    data, _sr = sf.read(str(AUDIO_FIXTURE), dtype="int16", always_2d=False)
    return data.tobytes()  # type: ignore[return-value]


def load_fixture_events(provider: str, scenario: str = "events-success") -> list[Any]:
    """Load JSON event list from ``fixtures/<provider>/<scenario>.json``."""
    path = FIXTURES_DIR / provider / f"{scenario}.json"
    return json.loads(path.read_text())  # type: ignore[return-value]
