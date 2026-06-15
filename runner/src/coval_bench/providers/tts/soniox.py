# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Soniox real-time TTS streaming provider."""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from uuid import uuid4

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_VALID_MODELS = ("tts-rt-v1",)
_VALID_VOICES = (
    "Maya",
    "Daniel",
    "Noah",
    "Nina",
    "Emma",
    "Jack",
    "Adrian",
    "Claire",
    "Grace",
    "Owen",
    "Mina",
    "Kenji",
    "Rafael",
    "Mateo",
    "Lucia",
    "Sofia",
    "Oliver",
    "Arthur",
    "Isla",
    "Victoria",
    "Cooper",
    "Mason",
    "Ruby",
    "Elise",
    "Arjun",
    "Rohan",
    "Priya",
    "Meera",
)
_WS_URL = "wss://tts-rt.soniox.com/tts-websocket"
_SAMPLE_RATE = 24000


class SonioxTTSProvider(TTSProvider):
    """Soniox TTS provider using WebSocket streaming (JSON frames, base64 audio)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Soniox TTS model {model!r}. Valid: {_VALID_MODELS}")
        if voice not in _VALID_VOICES:
            raise ValueError(f"Invalid Soniox TTS voice {voice!r}. Valid: {_VALID_VOICES}")
        self._model = model
        self._voice = voice

        api_key_secret = settings.soniox_api_key
        if api_key_secret is None:
            raise ValueError("soniox_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"soniox-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None
        stream_id = str(uuid4())

        try:
            async with ws_client.connect(_WS_URL) as ws:
                start = time.monotonic()
                # Soniox authenticates in-band: the api_key rides the opening config
                # frame rather than an Authorization header.
                await ws.send(
                    json.dumps(
                        {
                            "api_key": self._api_key,
                            "model": self._model,
                            "language": "en",
                            "voice": self._voice,
                            "audio_format": "pcm_s16le",
                            "sample_rate": _SAMPLE_RATE,
                            "stream_id": stream_id,
                        }
                    )
                )
                await ws.send(json.dumps({"text": text, "text_end": True, "stream_id": stream_id}))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)

                    if event.get("error_code") or event.get("error_message"):
                        message = event.get("error_message") or (
                            f"Soniox TTS error (code {event.get('error_code')})"
                        )
                        raise RuntimeError(str(message))

                    audio_b64 = event.get("audio")
                    if audio_b64:
                        chunk = base64.b64decode(audio_b64)
                        if chunk:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(chunk)

                    if event.get("terminated"):
                        break

        except Exception as exc:
            logger.debug("soniox_tts_error", exc_info=True)
            return finalize_tts_result(
                provider="soniox",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="soniox",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
