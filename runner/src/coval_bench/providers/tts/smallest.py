# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Smallest AI Lightning TTS provider — WebSocket streaming.

Wire protocol:
  connect → send JSON(text, voice_id, model, sample_rate)
  → recv JSON chunks: {"status": "chunk", "data": {"audio": "<base64-pcm>"}}
  → recv JSON done:   {"status": "complete", "done": true}

Auth:   Authorization: Bearer <key>
Output: 16-bit mono PCM (base64-decoded from JSON envelopes), 24 kHz
"""

from __future__ import annotations

import base64
import json
import time

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000
_WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"


class SmallestTTSProvider(TTSProvider):
    """Smallest AI Lightning TTS provider using WebSocket streaming."""

    _VALID_MODELS = frozenset({"lightning_v3.1_pro"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in self._VALID_MODELS:
            raise ValueError(
                f"Unsupported Smallest AI model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._model = model
        self._voice = voice

        api_key_secret = settings.smallest_api_key
        if api_key_secret is None:
            raise ValueError("smallest_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"smallest-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via Smallest AI WebSocket and return a TTSResult."""
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = json.dumps(
            {
                "text": text,
                "voice_id": self._voice,
                "model": self._model,
                "sample_rate": SAMPLE_RATE,
            }
        )

        try:
            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
                start = time.monotonic()
                await ws.send(payload)

                async for raw in ws:
                    if isinstance(raw, bytes):
                        # Graceful fallback: accept raw binary if server ever sends it.
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(raw)
                        continue

                    msg: dict = json.loads(raw)
                    status = msg.get("status", "")

                    if status == "chunk":
                        audio_b64: str = msg.get("data", {}).get("audio", "")
                        if audio_b64:
                            pcm = base64.b64decode(audio_b64)
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(pcm)
                    elif status == "complete" or msg.get("done"):
                        break

        except Exception as exc:
            logger.warning("smallest_error", exc_info=True)
            return finalize_tts_result(
                provider="smallest",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="smallest",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
