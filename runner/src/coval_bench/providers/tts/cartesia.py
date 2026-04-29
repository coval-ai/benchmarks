# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cartesia TTS provider — uses the official cartesia SDK over WebSocket."""

from __future__ import annotations

import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import structlog
from cartesia import AsyncCartesia
from cartesia.types import VoiceSpecifierParam

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000
OUTPUT_FORMAT: dict[str, Any] = {
    "sample_rate": SAMPLE_RATE,
    "container": "raw",
    "encoding": "pcm_s16le",
}


class CartesiaTTSProvider(TTSProvider):
    """Cartesia TTS provider using WebSocket streaming (cartesia SDK v2)."""

    enabled: bool = True

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

        api_key_secret = settings.cartesia_api_key
        if api_key_secret is None:
            raise ValueError("cartesia_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"cartesia-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via Cartesia WebSocket and return a TTSResult."""
        audio_chunks: list[bytes] = []
        ttfa_ms: float | None = None
        voice_spec: VoiceSpecifierParam = {"id": self._voice, "mode": "id"}

        try:
            client = AsyncCartesia(api_key=self._api_key)

            async with client.tts.websocket_connect() as conn:
                ctx = conn.context(
                    model_id=self._model,
                    voice=voice_spec,
                    output_format=OUTPUT_FORMAT,
                    language="en",
                )

                start = time.monotonic()
                await ctx.send(
                    model_id=self._model,
                    transcript=text,
                    voice=voice_spec,
                    continue_=False,
                )

                async for event in ctx.receive():
                    event_type: str = getattr(event, "type", "")
                    if event_type == "chunk":
                        audio: bytes | None = getattr(event, "audio", None)
                        if audio and len(audio) > 0:
                            if ttfa_ms is None:
                                ttfa_ms = (time.monotonic() - start) * 1000
                                logger.debug(
                                    "cartesia_ttfa",
                                    model=self._model,
                                    ttfa_ms=ttfa_ms,
                                )
                            audio_chunks.append(audio)
                    elif event_type == "done":
                        break

        except Exception as exc:
            logger.debug("cartesia_error", exc_info=True)
            return TTSResult(
                provider="cartesia",
                model=self._model,
                voice=self._voice,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_path = _write_wav(audio_chunks, SAMPLE_RATE) if audio_chunks else None
        return TTSResult(
            provider="cartesia",
            model=self._model,
            voice=self._voice,
            ttfa_ms=ttfa_ms,
            audio_path=audio_path,
            error=None,
        )


def _write_wav(chunks: list[bytes], sample_rate: int) -> Path:
    """Concatenate PCM chunks and write a WAV file to a temp location."""
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio_data = b"".join(chunks)
    with wave.open(tmp_name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return Path(tmp_name)
