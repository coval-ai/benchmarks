# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deepdub TTS provider — WebSocket streaming to the eTTS realtime API.

Wire protocol on wss://wsapi.deepdub.ai/open (``x-api-key`` header at the
handshake): send one ``text-to-speech`` request → recv JSON frames whose
``data`` field carries base64 PCM (an empty-data ack precedes audio) → the
final frame sets ``isFinished`` and may itself carry audio. Errors arrive as
frames with ``error``/``errorType``.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_WS_URL = "wss://wsapi.deepdub.ai/open"
_LOCALE = "en-US"

SAMPLE_RATE = 24000

# How many trailing non-audio frames to retain for the silent-stream diagnostic.
_LAST_FRAMES_KEPT = 3


def _raise_on_error(frame: dict[str, Any]) -> None:
    """Raise on a server error frame (``error``/``errorType``); others pass."""
    if frame.get("error") is not None:
        error_type = frame.get("errorType") or "error"
        raise RuntimeError(f"deepdub {error_type}: {frame['error']}")


class DeepdubTTSProvider(TTSProvider):
    """Deepdub TTS provider using the realtime WebSocket API (base64 PCM frames)."""

    _VALID_MODELS = frozenset({"dd-etts-3.0"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in self._VALID_MODELS:
            raise ValueError(
                f"Unsupported Deepdub model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if not voice:
            raise ValueError("Deepdub TTS requires a voice prompt id")
        self._model = model
        self._voice = voice

        api_key_secret = settings.deepdub_api_key
        if api_key_secret is None:
            raise ValueError("deepdub_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"deepdub-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        last_frames: list[str] = []
        start: float | None = None
        first_chunk_at: float | None = None

        request = {
            "action": "text-to-speech",
            "model": self._model,
            "targetText": text,
            "locale": _LOCALE,
            "voicePromptId": self._voice,
            "format": "s16le",
            "sampleRate": SAMPLE_RATE,
            "realtime": True,
        }

        try:
            async with ws_client.connect(
                _WS_URL, additional_headers={"x-api-key": self._api_key}
            ) as ws:
                start = time.monotonic()
                await ws.send(json.dumps(request))

                async for raw in ws:
                    text_frame = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                    try:
                        frame = json.loads(text_frame)
                    except json.JSONDecodeError:
                        last_frames.append(text_frame)
                        del last_frames[:-_LAST_FRAMES_KEPT]
                        continue

                    _raise_on_error(frame)
                    data = frame.get("data")
                    pcm = base64.b64decode(data) if data else b""
                    if pcm:
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(pcm)
                    else:
                        last_frames.append(text_frame)
                        del last_frames[:-_LAST_FRAMES_KEPT]
                    if frame.get("isFinished"):
                        break

        except Exception as exc:
            logger.warning("deepdub_tts_error", provider="deepdub", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="deepdub",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                last_frames=last_frames,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="deepdub",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
            last_frames=last_frames,
        )
