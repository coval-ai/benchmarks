# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deepgram TTS provider — WebSocket streaming to Deepgram Speak API.

Wire protocol (single-utterance benchmark path):
  connect → recv Metadata → send Speak(text) → send Flush
  → recv binary PCM frames until Flushed → close

Rate limit: 20 Flush messages per 60 seconds per API key.
"""

from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import urlencode

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000
_DEEPGRAM_TTS_WS_BASE = "wss://api.deepgram.com/v1/speak"


class DeepgramTTSProvider(TTSProvider):
    """Deepgram TTS provider using WebSocket streaming (Speak API)."""

    enabled: bool = True

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

        api_key_secret = settings.deepgram_api_key
        if api_key_secret is None:
            raise ValueError("deepgram_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"deepgram-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via Deepgram WebSocket and return a TTSResult."""
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        qs = urlencode({"encoding": "linear16", "sample_rate": SAMPLE_RATE, "model": self._model})
        url = f"{_DEEPGRAM_TTS_WS_BASE}?{qs}"
        headers = {"Authorization": f"Token {self._api_key}"}

        try:
            async with ws_client.connect(url, additional_headers=headers) as ws:
                # Drain the server-initiated Metadata frame before starting the clock.
                # Connection is fully established; t0 starts before Flush (synthesis trigger).
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    if msg.get("type") == "Warning":
                        logger.warning("deepgram_ws_warning", description=msg.get("description"))

                # t0 — synthesis trigger (equivalent to the HTTP POST in the old path).
                # Matches Cartesia's convention: t0 after connect, before text dispatch.
                start = time.monotonic()
                await ws.send(json.dumps({"type": "Speak", "text": text}))
                await ws.send(json.dumps({"type": "Flush"}))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(raw)
                        continue

                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")
                    if msg_type == "Flushed":
                        break
                    if msg_type == "Warning":
                        logger.warning("deepgram_ws_warning", description=msg.get("description"))

        except Exception as exc:
            logger.debug("deepgram_error", exc_info=True)
            return finalize_tts_result(
                provider="deepgram",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="deepgram",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
