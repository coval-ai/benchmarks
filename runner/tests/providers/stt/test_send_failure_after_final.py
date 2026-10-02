# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""A socket that dies mid-clip fails the clip, even after an earlier final (BENCH-1029)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt.assemblyai import AssemblyAIProvider
from coval_bench.providers.stt.azure import AzureSTTProvider
from coval_bench.providers.stt.baseten import BasetenSTTProvider
from coval_bench.providers.stt.cartesia import CartesiaSTTProvider
from coval_bench.providers.stt.cloudflare import CloudflareSTTProvider
from coval_bench.providers.stt.deepgram import DeepgramProvider
from coval_bench.providers.stt.elevenlabs import ElevenLabsSTTProvider
from coval_bench.providers.stt.gemini import GeminiSTTProvider
from coval_bench.providers.stt.gradium import GradiumSTTProvider
from coval_bench.providers.stt.guava import GuavaSTTProvider
from coval_bench.providers.stt.inworld import InworldSTTProvider
from coval_bench.providers.stt.mistral import MistralSTTProvider
from coval_bench.providers.stt.modulate import ModulateSTTProvider
from coval_bench.providers.stt.reson8 import Reson8STTProvider
from coval_bench.providers.stt.revai import RevAISTTProvider
from coval_bench.providers.stt.smallest import SmallestSTTProvider
from coval_bench.providers.stt.soniox import SonioxSTTProvider
from coval_bench.providers.stt.speechmatics import SpeechmaticsProvider
from coval_bench.providers.stt.together import TogetherSTTProvider
from coval_bench.providers.stt.zoom import ZoomSTTProvider

_KEY = SecretStr("test-key")
_PCM = b"\x01\x02" * 1600

_HANDSHAKES = (
    "_wait_for_setup_complete",
    "_await_session_created",
    "_wait_for_recognition_started",
)

_PROVIDERS: dict[str, Any] = {
    "assemblyai": lambda: AssemblyAIProvider(_KEY),
    "azure": lambda: AzureSTTProvider(_KEY, region="eastus"),
    "baseten": lambda: BasetenSTTProvider(_KEY, ws_url="wss://baseten.example/ws"),
    "cartesia": lambda: CartesiaSTTProvider(_KEY),
    "cloudflare": lambda: CloudflareSTTProvider(_KEY, "nova-3", "example-account"),
    "deepgram": lambda: DeepgramProvider(_KEY, "nova-3"),
    "elevenlabs": lambda: ElevenLabsSTTProvider(_KEY),
    "gemini": lambda: GeminiSTTProvider(_KEY),
    "gradium": lambda: GradiumSTTProvider(_KEY),
    "guava": lambda: GuavaSTTProvider(_KEY, base_url="https://guava.example"),
    "inworld": lambda: InworldSTTProvider(_KEY),
    "mistral": lambda: MistralSTTProvider(_KEY),
    "modulate": lambda: ModulateSTTProvider(_KEY),
    "reson8": lambda: Reson8STTProvider(_KEY),
    "revai": lambda: RevAISTTProvider(_KEY),
    "smallest": lambda: SmallestSTTProvider(_KEY),
    "soniox": lambda: SonioxSTTProvider(_KEY),
    "speechmatics": lambda: SpeechmaticsProvider(_KEY),
    "together": lambda: TogetherSTTProvider(_KEY),
    "zoom": lambda: ZoomSTTProvider(_KEY, SecretStr("test-secret-at-least-32-bytes-long")),
}


async def _final_arrives(self: Any, ws: Any, result: TranscriptionResult, *_: Any) -> None:
    result.complete_transcript = "the first sentence"
    result.audio_to_final_seconds = 1.0


def _connect() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.recv = AsyncMock(return_value=json.dumps({"message_type": "session_started"}))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


async def _measure(provider: STTProvider, send_audio: Any) -> TranscriptionResult:
    cls = type(provider)
    module = cls.__module__
    patches = [
        patch.object(cls, "_send_audio", send_audio),
        patch.object(cls, "_receive", _final_arrives),
        patch(f"{module}.ws_client.connect", return_value=_connect()),
        *(patch.object(cls, name, AsyncMock()) for name in _HANDSHAKES if hasattr(cls, name)),
    ]
    for p in patches:
        p.start()
    try:
        return await provider.measure_ttft(_PCM, 1, 2, 16000, 0.1)
    finally:
        for p in reversed(patches):
            p.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(_PROVIDERS))
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

    result = await _measure(_PROVIDERS[name](), dies_mid_clip)

    assert result.error is not None
    assert result.error == str(closed)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(_PROVIDERS))
async def test_full_clip_with_final_stays_success(name: str) -> None:
    async def sends_everything(*_: Any) -> None:
        return None

    result = await _measure(_PROVIDERS[name](), sends_everything)

    assert result.error is None
    assert result.complete_transcript == "the first sentence"
