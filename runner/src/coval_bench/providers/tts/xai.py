# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""xAI Grok TTS streaming provider."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult

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
            "text_normalization": "true",
        }
        return f"{_BASE_WS_URL}?{urlencode(params)}"

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        ttfa_ms: float | None = None

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
                                if ttfa_ms is None:
                                    ttfa_ms = (time.monotonic() - start) * 1000
                                audio_chunks.append(chunk)

                    elif event_type == "audio.done":
                        break

                    elif event_type == "error":
                        raise RuntimeError(str(event.get("message", "xAI TTS error")))

        except Exception as exc:
            logger.debug("xai_tts_error", exc_info=True)
            return TTSResult(
                provider="xai",
                model=self._model,
                voice=self._voice,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_path = _write_wav(audio_chunks, _SAMPLE_RATE) if audio_chunks else None
        return TTSResult(
            provider="xai",
            model=self._model,
            voice=self._voice,
            ttfa_ms=ttfa_ms,
            audio_path=audio_path,
            error=None,
        )


def _write_wav(chunks: list[bytes], sample_rate: int) -> Path:
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio_data = b"".join(chunks)
    with wave.open(tmp_name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return Path(tmp_name)
