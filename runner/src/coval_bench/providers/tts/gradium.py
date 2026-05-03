# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradium TTS provider — WebSocket streaming.

Wire protocol: WebSocket, wss://api.gradium.ai/api/speech/tts
Auth: x-api-key: <key>
Setup: {"type":"setup","voice_id":"...","model_name":"default","output_format":"pcm"}
Text:  {"type":"text","text":"..."}
Close: {"type":"end_of_stream"}
Audio: server sends {"type":"audio","audio":"<base64 PCM>"} chunks
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_VALID_MODELS = ("default",)
_WS_URL = "wss://api.gradium.ai/api/speech/tts"
_SAMPLE_RATE = 24000


class GradiumTTSProvider(TTSProvider):
    """Gradium TTS provider using WebSocket streaming."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Gradium TTS model {model!r}. Valid: {_VALID_MODELS}")
        self._model = model
        self._voice = voice

        api_key_secret = settings.gradium_tts_api_key
        if api_key_secret is None:
            raise ValueError("gradium_tts_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return "gradium"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        ttfa_ms: float | None = None

        try:
            headers = {"x-api-key": self._api_key}

            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "setup",
                            "voice_id": self._voice,
                            "model_name": self._model,
                            "output_format": "pcm",
                        }
                    )
                )

                # Wait for ready
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg: dict[str, Any] = json.loads(raw)
                if msg.get("type") != "ready":
                    logger.warning("gradium unexpected first message", msg=msg)

                start = time.monotonic()
                await ws.send(json.dumps({"type": "text", "text": text}))
                await ws.send(json.dumps({"type": "end_of_stream"}))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    msg_type: str = msg.get("type", "")

                    if msg_type == "audio":
                        audio_b64: str = msg.get("audio", "")
                        if audio_b64:
                            chunk = base64.b64decode(audio_b64)
                            if chunk:
                                if ttfa_ms is None:
                                    ttfa_ms = (time.monotonic() - start) * 1000
                                    logger.debug(
                                        "gradium_ttfa",
                                        model=self._model,
                                        ttfa_ms=ttfa_ms,
                                    )
                                audio_chunks.append(chunk)

                    elif msg_type == "end_of_stream":
                        break

                    elif msg_type == "error":
                        raise RuntimeError(str(msg.get("message", msg)))

        except Exception as exc:
            logger.debug("gradium_tts_error", exc_info=True)
            return TTSResult(
                provider="gradium",
                model=self._model,
                voice=self._voice,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_path = _write_wav(audio_chunks, _SAMPLE_RATE) if audio_chunks else None
        return TTSResult(
            provider="gradium",
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
