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
import os
import tempfile
import time
import wave
from pathlib import Path
from urllib.parse import urlencode

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult

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

    def _model_supported(self, model: str) -> bool:
        return model.startswith("aura-")

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via Deepgram WebSocket and return a TTSResult."""
        if not self._model_supported(self._model):
            return TTSResult(
                provider="deepgram",
                model=self._model,
                voice=self._voice,
                ttfa_ms=None,
                audio_path=None,
                error=(
                    f"Unsupported Deepgram TTS model: {self._model}. Expected an 'aura-' model."
                ),
            )
        audio_chunks: list[bytes] = []
        ttfa_ms: float | None = None

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

                await ws.send(json.dumps({"type": "Speak", "text": text}))

                # t0 — synthesis trigger (equivalent to the HTTP POST in the old path).
                # Matches Cartesia's convention: t0 after connect, before text dispatch.
                start = time.monotonic()
                await ws.send(json.dumps({"type": "Flush"}))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        if ttfa_ms is None:
                            ttfa_ms = (time.monotonic() - start) * 1000
                            logger.debug(
                                "deepgram_ttfa",
                                model=self._model,
                                ttfa_ms=ttfa_ms,
                            )
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
            return TTSResult(
                provider="deepgram",
                model=self._model,
                voice=self._voice,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_path = _write_wav(audio_chunks, SAMPLE_RATE) if audio_chunks else None
        return TTSResult(
            provider="deepgram",
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
