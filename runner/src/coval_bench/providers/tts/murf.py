# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Murf TTS provider — WebSocket streaming via the stream-input API."""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from urllib.parse import urlencode

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_VALID_MODELS = ("falcon-2",)
_WS_URL = "wss://us-east.api.murf.ai/v1/speech/stream-input"
_SAMPLE_RATE = 24000


class MurfTTSProvider(TTSProvider):
    """Murf TTS provider using WebSocket streaming (JSON frames, base64 audio)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Murf TTS model {model!r}. Valid: {_VALID_MODELS}")
        if not voice:
            raise ValueError("Murf TTS requires a voice")
        self._model = model
        self._voice = voice

        api_key_secret = settings.murfai_api_key
        if api_key_secret is None:
            raise ValueError("murfai_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"murf-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        query = urlencode(
            {
                "api-key": self._api_key,
                "model": self._model,
                "sample_rate": _SAMPLE_RATE,
                "channel_type": "MONO",
                "format": "PCM",
            }
        )

        try:
            async with ws_client.connect(f"{_WS_URL}?{query}") as ws:
                await ws.send(
                    json.dumps({"voice_config": {"voiceId": self._voice, "locale": "en-US"}})
                )

                start = time.monotonic()
                await ws.send(json.dumps({"text": text, "end": True}))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)

                    error = event.get("error") or event.get("errorMessage")
                    if error:
                        raise RuntimeError(str(error))

                    audio_b64 = event.get("audio", "")
                    if audio_b64:
                        chunk = base64.b64decode(audio_b64)
                        if chunk:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(chunk)

                    if event.get("final"):
                        break

        except Exception as exc:
            logger.warning("murf_tts_error", provider="murf", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="murf",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="murf",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
