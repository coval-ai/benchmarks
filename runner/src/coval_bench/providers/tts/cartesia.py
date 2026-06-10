# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cartesia TTS provider — uses the official cartesia SDK over WebSocket."""

from __future__ import annotations

import time
from typing import Any

import structlog
from cartesia import AsyncCartesia
from cartesia.types import VoiceSpecifierParam

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000
OUTPUT_FORMAT: dict[str, Any] = {
    "sample_rate": SAMPLE_RATE,
    "container": "raw",
    "encoding": "pcm_s16le",
}


class CartesiaTTSProvider(TTSProvider):
    """Cartesia TTS provider using WebSocket streaming (cartesia SDK v2)."""

    _VALID_MODELS = frozenset({"sonic-3", "sonic-3.5", "sonic-turbo"})

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
        if not self._model_supported(self._model):
            return TTSResult(
                provider="cartesia",
                model=self._model,
                voice=self._voice,
                ttfa_ms=None,
                audio_path=None,
                error=(
                    f"Unsupported Cartesia model: {self._model}. "
                    f"Valid models: {sorted(self._VALID_MODELS)}"
                ),
            )
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None
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
                # Both `output_format` and `language` MUST be passed on every
                # send() — ctx.send() does NOT inherit them from conn.context().
                # Omitting output_format causes the SDK to substitute its own
                # default (pcm_f32le / 44100 Hz), which mismatches the WAV header
                # written at finalize (pcm_s16le / 24000 Hz) and produces
                # corrupt audio → Whisper hallucination → WER > 100%.
                # Omitting language causes sonic-3 to auto-detect and intermittently
                # synthesize non-English audio → WER = 100%.
                await ctx.send(
                    model_id=self._model,
                    transcript=text,
                    voice=voice_spec,
                    output_format=OUTPUT_FORMAT,
                    language="en",
                    continue_=False,
                )

                async for event in ctx.receive():
                    event_type: str = getattr(event, "type", "")
                    if event_type == "chunk":
                        audio: bytes | None = getattr(event, "audio", None)
                        if audio and len(audio) > 0:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(audio)
                    elif event_type == "done":
                        break

        except Exception as exc:
            logger.debug("cartesia_error", exc_info=True)
            return finalize_tts_result(
                provider="cartesia",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="cartesia",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
