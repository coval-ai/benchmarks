# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""WebSocket fakes shared by the STT and TTS provider tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

Event = dict[str, Any] | str | bytes


class FakeWebSocket:
    """Async context-manager WebSocket fake that replays events from a list.

    ``dict`` events are sent as JSON; ``str`` and ``bytes`` go out verbatim.
    ``recv()`` and ``async for`` drain the same list. With ``server_closes=False``
    iteration blocks until the client closes, modelling a server (e.g. Inworld)
    that keeps the socket open. Everything the client sends lands in ``sent``.
    """

    def __init__(
        self,
        events: Sequence[Event],
        on_send: Callable[[Any], None] | None = None,
        *,
        server_closes: bool = True,
    ) -> None:
        self.events: list[Event] = list(events)
        self.sent: list[Any] = []
        self.closed = asyncio.Event()
        self._on_send = on_send
        self._server_closes = server_closes

    async def __aenter__(self) -> FakeWebSocket:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed.set()

    async def close(self) -> None:
        self.closed.set()
        self.events.clear()

    async def send(self, msg: bytes | str) -> None:
        if self.closed.is_set():
            raise ConnectionError("send on a closed FakeWebSocket")
        self.sent.append(msg)
        if self._on_send is not None:
            self._on_send(msg)

    async def recv(self) -> str | bytes:
        if not self.events:
            raise StopAsyncIteration
        return _encode(self.events.pop(0))

    def __aiter__(self) -> FakeWebSocket:
        return self

    async def __anext__(self) -> str | bytes:
        while not self.events:
            if self._server_closes or self.closed.is_set():
                raise StopAsyncIteration
            await self.closed.wait()
        return _encode(self.events.pop(0))


def _encode(event: Event) -> str | bytes:
    return json.dumps(event) if isinstance(event, dict) else event


def fake_connect(
    events: Sequence[Event] | object,
    sent: list[Any] | None = None,
    *,
    server_closes: bool = True,
) -> Any:
    """A ``ws_client.connect`` stand-in: an async CM yielding a socket, also exposed as ``.ws``.

    Pass an event list to get a FakeWebSocket, or any ready-made socket fake to wrap it.
    """
    ws = (
        FakeWebSocket(
            events, on_send=sent.append if sent is not None else None, server_closes=server_closes
        )
        if isinstance(events, list | tuple)
        else events
    )
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    cm.ws = ws
    return cm
