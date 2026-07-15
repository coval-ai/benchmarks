# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Fish Audio TTS provider (S1 / S2 line over WebSocket).

Protocol: MessagePack-serialized events on ``wss://api.fish.audio/v1/tts/live``.
The model is chosen by a ``model`` header on the WS handshake; the voice is a
``reference_id`` from the Fish Audio voice library.
"""

from __future__ import annotations

import time
from typing import Any

import ormsgpack
import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_VALID_MODELS = ("s1", "s2.1-pro", "s2.1-pro-free")
_WS_URL = "wss://api.fish.audio/v1/tts/live"
_SAMPLE_RATE = 44100
_MAX_WS_SIZE = 16 * 1024 * 1024


class FishAudioTTSProvider(TTSProvider):
    """Fish Audio TTS provider using WebSocket streaming."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Fish Audio TTS model {model!r}. Valid: {_VALID_MODELS}")
        if not voice:
            raise ValueError("Fish Audio TTS requires a voice (library reference_id)")
        self._model = model
        self._voice = voice

        api_key_secret = settings.fishaudio_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value().strip():
            raise ValueError("fishaudio_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"fishaudio-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None
        headers = {"Authorization": f"Bearer {self._api_key}", "model": self._model}

        try:
            async with ws_client.connect(
                _WS_URL,
                additional_headers=headers,
                max_size=_MAX_WS_SIZE,
            ) as ws:
                # Clock starts post-handshake so TTFA excludes connect (cohort parity).
                start = time.monotonic()
                # latency "balanced" streams chunks as synthesized; the default
                # "normal" buffers whole segments, folding generation into TTFA.
                await ws.send(
                    ormsgpack.packb(
                        {
                            "event": "start",
                            "request": {
                                "text": "",
                                "format": "pcm",
                                "sample_rate": _SAMPLE_RATE,
                                "reference_id": self._voice,
                                "latency": "balanced",
                                "features": ["quality-guard"],
                            },
                        }
                    )
                )
                await ws.send(ormsgpack.packb({"event": "text", "text": text}))
                await ws.send(ormsgpack.packb({"event": "stop"}))

                async for message in ws:
                    if not isinstance(message, (bytes, bytearray)):
                        continue
                    data: dict[str, Any] = ormsgpack.unpackb(bytes(message))
                    event = data.get("event")
                    if event == "audio":
                        chunk = data.get("audio")
                        if chunk:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(chunk)
                    elif event == "finish":
                        if data.get("reason") == "error":
                            raise RuntimeError(str(data.get("message", "finish reason=error")))
                        break

        except Exception as exc:
            logger.warning(
                "fishaudio_tts_error", provider="fishaudio", model=self._model, exc_info=exc
            )
            return finalize_tts_result(
                provider="fishaudio",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="fishaudio",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
