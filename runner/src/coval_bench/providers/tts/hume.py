# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hume TTS provider — WebSocket streaming to Hume Octave TTS API.
Hume WS does not accept headers
"""

from __future__ import annotations

import asyncio
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

SAMPLE_RATE = 48000
_WS_BASE = "wss://api.hume.ai/v0/tts/stream/input"
_WS_SESSION_TIMEOUT_S = 30.0

# Maps model name to Hume version query param
_MODEL_TO_VERSION: dict[str, str] = {"octave-tts": "1", "octave-2": "2"}

_DEFAULT_VOICE_ID = "176a55b1-4468-4736-8878-db82729667c1"

# How many trailing text frames to retain for the silent-stream diagnostic.
_LAST_FRAMES_KEPT = 3


def _frame_error(msg: dict[str, Any]) -> str | None:
    """Hume's error text for an error frame, or ``None`` when the frame is benign.

    Hume nests the discriminator under ``details`` and sends no top-level ``type``,
    so the original top-level check never matched and every error frame was dropped —
    including "Exhausted credit balance", stated on every request for four and a half
    days. ``status_code`` is checked too so an error shape we haven't seen still
    registers rather than being silently ignored.
    """
    details = msg.get("details")
    nested_type = details.get("type") if isinstance(details, dict) else None
    status_code = msg.get("status_code")
    is_error = (
        msg.get("type") == "error"
        or nested_type == "error"
        or (isinstance(status_code, int) and status_code >= 400)
    )
    if not is_error:
        return None
    return str(msg.get("message") or msg)


class HumeTTSProvider(TTSProvider):
    """Hume TTS provider using WebSocket streaming (Octave Speak API)."""

    _VALID_MODELS = frozenset(_MODEL_TO_VERSION)

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
        if not self._model_supported(self._model):
            return TTSResult(
                provider="hume",
                model=self._model,
                voice=voice_id,
                ttfa_ms=None,
                audio_path=None,
                error=(
                    f"Unsupported Hume model: {self._model}. "
                    f"Supported: {sorted(self._VALID_MODELS)}"
                ),
            )

        version = _MODEL_TO_VERSION[self._model]

        audio_chunks: list[bytes] = []
        last_frames: list[str] = []
        start: float | None = None
        first_chunk_at: float | None = None

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
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(raw)
                        elif isinstance(raw, str):
                            last_frames.append(raw)
                            del last_frames[:-_LAST_FRAMES_KEPT]
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            failure = _frame_error(msg)
                            if failure is not None:
                                raise RuntimeError(failure)

        except TimeoutError:
            logger.warning(
                "hume_timeout",
                provider="hume",
                model=self._model,
                timeout_s=_WS_SESSION_TIMEOUT_S,
            )
            return finalize_tts_result(
                provider="hume",
                model=self._model,
                voice=voice_id,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                last_frames=last_frames,
                error=f"Hume WebSocket session timed out after {_WS_SESSION_TIMEOUT_S}s",
            )

        except Exception as exc:
            logger.warning("hume_error", provider="hume", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="hume",
                model=self._model,
                voice=voice_id,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                last_frames=last_frames,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="hume",
            model=self._model,
            voice=voice_id,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
            last_frames=last_frames,
        )
