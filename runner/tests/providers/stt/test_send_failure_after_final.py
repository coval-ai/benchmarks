# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""A socket that dies mid-clip fails the clip, even after an earlier final (BENCH-1029)."""

from __future__ import annotations

import inspect
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt import STT_PROVIDERS
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import make_settings

_PCM = b"\x01\x02" * 1600

# Every websocket provider; google is gRPC and has no send/receive split.
_PROVIDERS = sorted(
    name for name, cls in STT_PROVIDERS.items() if issubclass(cls, WebSocketSTTProvider)
)
_MODELS = {"cloudflare": "nova-3", "deepgram": "nova-3", "stepfun": "step-asr"}


def _provider(name: str) -> WebSocketSTTProvider:
    cls = STT_PROVIDERS[name]
    model = _MODELS.get(name) or inspect.signature(cls.__init__).parameters["model"].default
    provider = cls(make_settings(), model)
    assert isinstance(provider, WebSocketSTTProvider)
    return provider


async def _final_arrives(self: Any, ws: Any, tx: Transcription) -> None:
    tx.result.audio_start_time = time.monotonic() - 1.0
    tx.final("the first sentence", time.monotonic())


def _connect() -> Any:
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return fake_connect(ws)


async def _measure(provider: WebSocketSTTProvider, send: Any) -> TranscriptionResult:
    cls = type(provider)
    with (
        patch.object(cls, "_connect", AsyncMock(return_value=_connect())),
        patch.object(cls, "_open", AsyncMock()),
        patch.object(cls, "_send", send),
        patch.object(cls, "_receive", _final_arrives),
    ):
        return await provider.measure_ttft(_PCM, 1, 2, 16000, 0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _PROVIDERS)
@pytest.mark.parametrize(
    "closed",
    [
        ConnectionClosedError(
            Close(4503, "no_healthy_workers"), Close(4503, "no_healthy_workers"), True
        ),
        ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True),
    ],
    ids=["abnormal-4503", "normal-1000"],
)
async def test_socket_dies_mid_clip_after_final_fails_the_clip(
    name: str, closed: Exception
) -> None:
    async def dies_mid_clip(*_: Any) -> None:
        raise closed

    result = await _measure(_provider(name), dies_mid_clip)

    assert result.error is not None
    assert result.error == str(closed)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _PROVIDERS)
async def test_clean_send_after_final_keeps_the_clip(name: str) -> None:
    async def sends_fine(*_: Any) -> None:
        return None

    result = await _measure(_provider(name), sends_fine)

    assert result.error is None
