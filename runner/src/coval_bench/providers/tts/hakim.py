# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hakim AI TTS provider — WebSocket streaming via the realtime speech API.

Wire protocol on wss://api.tryhakim.ai/v1/audio/speech/stream (Bearer auth at
the handshake): send one ``speech.create`` → audio arrives as binary
PCM-S16LE frames (24 kHz mono) between the matching ``speech.started`` /
``speech.done`` JSON frames. Errors arrive as ``error`` frames with a stable
``code``; lifecycle frames (``session.created``, ``session.usage``) interleave
and carry no audio.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_VALID_MODELS = ("hakim-fast-v1",)
_WS_URL = "wss://api.tryhakim.ai/v1/audio/speech/stream"
_SAMPLE_RATE = 24000

# How many trailing non-audio frames to retain for the silent-stream diagnostic.
_LAST_FRAMES_KEPT = 3


class HakimTTSProvider(TTSProvider):
    """Hakim AI TTS provider using the realtime WebSocket API (binary PCM frames)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Hakim TTS model {model!r}. Valid: {_VALID_MODELS}")
        if not voice:
            raise ValueError("Hakim TTS requires a voice")
        self._model = model
        self._voice = voice

        api_key_secret = settings.hakimai_api_key
        if api_key_secret is None:
            raise ValueError("hakimai_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"hakim-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        last_frames: list[str] = []
        start: float | None = None
        first_chunk_at: float | None = None
        done = False

        request = {
            "type": "speech.create",
            "input": text,
            "model": self._model,
            "voice": self._voice,
        }

        try:
            async with ws_client.connect(
                _WS_URL, additional_headers={"Authorization": f"Bearer {self._api_key}"}
            ) as ws:
                start = time.monotonic()
                await ws.send(json.dumps(request))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        if raw:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(raw)
                        continue

                    frame: dict[str, Any] = json.loads(raw)
                    if frame.get("type") == "error":
                        code = frame.get("code") or "error"
                        raise RuntimeError(f"hakim {code}: {frame.get('message')}")
                    last_frames.append(raw)
                    del last_frames[:-_LAST_FRAMES_KEPT]
                    if frame.get("type") == "speech.done":
                        done = True
                        break

                # A clean close before ``speech.done`` is a truncated stream; audio
                # collected so far must not be scored as a complete synthesis.
                if not done:
                    raise RuntimeError("connection closed before the speech.done frame")

        except Exception as exc:
            logger.warning("hakim_tts_error", provider="hakim", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="hakim",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                last_frames=last_frames,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="hakim",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
            last_frames=last_frames,
        )
