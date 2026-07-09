# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Alibaba Cloud Model Studio TTS provider (Qwen3-TTS-Flash over the DashScope realtime API)."""

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

_DEFAULT_WS_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
_VALID_MODELS = ("qwen3-tts-flash-realtime",)
_VALID_VOICES = ("Cherry", "Ethan")
_SAMPLE_RATE = 24000
_MAX_WS_SIZE = 16 * 1024 * 1024


class AlibabaTTSProvider(TTSProvider):
    """Alibaba Cloud TTS provider using the DashScope realtime WebSocket (Qwen3-TTS-Flash)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Alibaba TTS model {model!r}. Valid: {_VALID_MODELS}")
        if voice not in _VALID_VOICES:
            raise ValueError(f"Invalid Alibaba TTS voice {voice!r}. Valid: {_VALID_VOICES}")
        self._model = model
        self._voice = voice

        api_key_secret = settings.alibaba_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value().strip():
            raise ValueError("alibaba_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

        base_url = settings.alibaba_tts_url or _DEFAULT_WS_URL
        self._ws_url = f"{base_url}?model={model}"

    @property
    def name(self) -> str:
        return "alibaba"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with ws_client.connect(
                self._ws_url,
                additional_headers=headers,
                max_size=_MAX_WS_SIZE,
            ) as ws:
                start = time.monotonic()
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "voice": self._voice,
                                "mode": "commit",
                                "response_format": "pcm",
                                "sample_rate": _SAMPLE_RATE,
                            },
                        }
                    )
                )
                await ws.send(json.dumps({"type": "input_text_buffer.append", "text": text}))
                await ws.send(json.dumps({"type": "input_text_buffer.commit"}))
                await ws.send(json.dumps({"type": "session.finish"}))

                async for message in ws:
                    if isinstance(message, (bytes, bytearray)):
                        continue
                    data: dict[str, Any] = json.loads(message)
                    event_type = data.get("type")
                    if event_type == "response.audio.delta":
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(base64.b64decode(data.get("delta", "")))
                    elif event_type == "session.finished":
                        break
                    elif event_type == "error":
                        raise RuntimeError(str(data.get("error", data)))

        except Exception as exc:
            logger.warning("alibaba_tts_error", provider="alibaba", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="alibaba",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="alibaba",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
