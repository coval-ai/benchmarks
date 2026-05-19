# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hume TTS provider — WebSocket streaming to Hume Octave TTS API.
Hume WS does not accept headers
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

SAMPLE_RATE = 48000
_WS_BASE = "wss://api.hume.ai/v0/tts/stream/input"
_WS_SESSION_TIMEOUT_S = 30.0

SUPPORTED_MODELS = {"octave-tts", "octave-2"}
# Maps model name to Hume version query param
_MODEL_TO_VERSION: dict[str, str] = {"octave-tts": "1", "octave-2": "2"}

_DEFAULT_VOICE_ID = "176a55b1-4468-4736-8878-db82729667c1"


class HumeTTSProvider(TTSProvider):
    """Hume TTS provider using WebSocket streaming (Octave Speak API)."""

    enabled: bool = False

    def __init__(self, settings: Settings, model: str, voice: str | None) -> None:
        self._model = model
        self._voice = voice

        api_key_secret = settings.hume_api_key
        if api_key_secret is None:
            raise ValueError("hume_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"hume-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via Hume WebSocket and return a TTSResult."""
        voice_id = self._voice if self._voice else _DEFAULT_VOICE_ID
        if self._model not in SUPPORTED_MODELS:
            return TTSResult(
                provider="hume",
                model=self._model,
                voice=voice_id,
                ttfa_ms=None,
                audio_path=None,
                error=(
                    f"Unsupported Hume model: {self._model}. Supported: {sorted(SUPPORTED_MODELS)}"
                ),
            )

        version = _MODEL_TO_VERSION[self._model]

        audio_chunks: list[bytes] = []
        ttfa_ms: float | None = None

        qs = urlencode(
            {
                "api_key": self._api_key,
                "instant_mode": "true",
                "format_type": "pcm",
                "strip_headers": "true",
                "version": version,
            }
        )
        url = f"{_WS_BASE}?{qs}"

        try:
            async with asyncio.timeout(_WS_SESSION_TIMEOUT_S):
                async with ws_client.connect(url) as ws:
                    # t0: connection established; Hume WS sends no setup message.
                    start = time.monotonic()

                    await ws.send(
                        json.dumps(
                            {
                                "text": text,
                                "voice": {"id": voice_id, "provider": "HUME_AI"},
                                "speed": 1.0,
                                "trailing_silence": 0,
                            }
                        )
                    )
                    await ws.send(json.dumps({"flush": True}))
                    await ws.send(json.dumps({"close": True}))

                    async for raw in ws:
                        if isinstance(raw, bytes) and len(raw) > 0:
                            if ttfa_ms is None:
                                ttfa_ms = (time.monotonic() - start) * 1000
                                logger.debug(
                                    "hume_ttfa",
                                    model=self._model,
                                    ttfa_ms=ttfa_ms,
                                )
                            audio_chunks.append(raw)
                        elif isinstance(raw, str):
                            try:
                                msg = json.loads(raw)
                                if msg.get("type") == "error":
                                    raise RuntimeError(str(msg.get("message", msg)))
                            except (json.JSONDecodeError, KeyError):
                                pass

        except TimeoutError:
            logger.warning(
                "hume_timeout",
                model=self._model,
                timeout_s=_WS_SESSION_TIMEOUT_S,
            )
            return TTSResult(
                provider="hume",
                model=self._model,
                voice=voice_id,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=f"Hume WebSocket session timed out after {_WS_SESSION_TIMEOUT_S}s",
            )

        except Exception as exc:
            logger.warning("hume_error", exc_info=True)
            return TTSResult(
                provider="hume",
                model=self._model,
                voice=voice_id,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_path = _write_wav(audio_chunks, SAMPLE_RATE) if audio_chunks else None
        return TTSResult(
            provider="hume",
            model=self._model,
            voice=voice_id,
            ttfa_ms=ttfa_ms,
            audio_path=audio_path,
            error=None,
        )


def _write_wav(chunks: list[bytes], sample_rate: int) -> Path:
    """Concatenate raw PCM chunks and write a WAV file."""
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio_data = b"".join(chunks)
    with wave.open(tmp_name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return Path(tmp_name)
