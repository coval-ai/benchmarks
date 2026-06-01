# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""xAI Grok TTS streaming provider."""

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

_VALID_MODELS = ("grok-tts",)
_VALID_VOICES = ("eve", "ara", "rex", "sal", "leo")
_BASE_WS_URL = "wss://api.x.ai/v1/tts"
_SAMPLE_RATE = 24000


class XaiTTSProvider(TTSProvider):
    """xAI Grok TTS provider using WebSocket streaming."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid xAI TTS model {model!r}. Valid: {_VALID_MODELS}")
        if voice not in _VALID_VOICES:
            raise ValueError(f"Invalid xAI TTS voice {voice!r}. Valid: {_VALID_VOICES}")
        self._model = model
        self._voice = voice

        api_key_secret = settings.xai_api_key
        if api_key_secret is None:
            raise ValueError("xai_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"xai-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self) -> str:
        params: dict[str, str | int] = {
            "language": "en",
            "voice": self._voice,
            "codec": "pcm",
            "sample_rate": _SAMPLE_RATE,
            "text_normalization": "false",
            "optimize_streaming_latency": 2,
        }
        return f"{_BASE_WS_URL}?{urlencode(params)}"

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        try:
            headers = {"Authorization": f"Bearer {self._api_key}"}

            async with ws_client.connect(
                self._build_websocket_url(),
                additional_headers=headers,
            ) as ws:
                start = time.monotonic()
                await ws.send(json.dumps({"type": "text.delta", "delta": text}))
                await ws.send(json.dumps({"type": "text.done"}))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)
                    event_type = str(event.get("type", ""))

                    if event_type == "audio.delta":
                        audio_b64 = str(event.get("delta", ""))
                        if audio_b64:
                            chunk = base64.b64decode(audio_b64)
                            if chunk:
                                if first_chunk_at is None:
                                    first_chunk_at = time.monotonic()
                                audio_chunks.append(chunk)

                    elif event_type == "audio.done":
                        break

                    elif event_type == "error":
                        raise RuntimeError(str(event.get("message", "xAI TTS error")))

        except Exception as exc:
            logger.debug("xai_tts_error", exc_info=True)
            return finalize_tts_result(
                provider="xai",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="xai",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
