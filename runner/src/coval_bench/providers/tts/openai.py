# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI TTS provider — supports HTTP streaming and Realtime WebSocket modes."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import structlog
import websockets
import websockets.exceptions
from openai import AsyncOpenAI

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

VALID_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]

HTTP_MODELS = {"gpt-4o-mini-tts", "tts-1", "tts-1-hd"}
REALTIME_MODELS = {"gpt-realtime-2025-08-28"}
SAMPLE_RATE = 24000


class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS provider supporting HTTP and Realtime WS paths."""

    enabled: bool = True

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice
        self._settings = settings

        if self._voice not in VALID_VOICES:
            logger.warning(
                "unknown_openai_voice",
                voice=self._voice,
                fallback="alloy",
            )
            self._voice = "alloy"

        api_key_secret = settings.openai_api_key
        if api_key_secret is None:
            raise ValueError("openai_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()
        self._client = AsyncOpenAI(api_key=self._api_key)

    @property
    def name(self) -> str:
        return f"openai-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech and return a TTSResult."""
        if self._model in HTTP_MODELS:
            return await self._synthesize_http(text)
        if self._model in REALTIME_MODELS:
            return await self._synthesize_realtime(text)
        return TTSResult(
            provider="openai",
            model=self._model,
            voice=self._voice,
            ttfa_ms=None,
            audio_path=None,
            error=f"Unsupported OpenAI model: {self._model}",
        )

    async def _synthesize_http(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        try:
            start = time.monotonic()
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=self._voice,
                input=text,
                response_format="pcm",
            ) as response:
                async for chunk in response.iter_bytes():
                    if isinstance(chunk, bytes) and len(chunk) > 0:
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(chunk)
        except Exception as exc:
            logger.debug("openai_http_error", exc_info=True)
            return finalize_tts_result(
                provider="openai",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="openai",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )

    async def _synthesize_realtime(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        ws_url = f"wss://api.openai.com/v1/realtime?model={self._model}"
        ws_extra_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        create_event: dict[str, Any] = {
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "instructions": f"Speak this text exactly as provided: {text}",
                "voice": self._voice,
                "output_audio_format": "pcm16",
            },
        }

        try:
            start = time.monotonic()
            async with websockets.connect(ws_url, additional_headers=ws_extra_headers) as ws:
                await ws.send(json.dumps(create_event))
                while True:
                    try:
                        raw = await ws.recv()
                        event: dict[str, Any] = json.loads(raw)
                        event_type: str = event.get("type", "")

                        if event_type == "response.audio.delta" and "delta" in event:
                            audio_data = base64.b64decode(event["delta"])
                            if len(audio_data) > 0:
                                if first_chunk_at is None:
                                    first_chunk_at = time.monotonic()
                                audio_chunks.append(audio_data)

                        if event_type == "response.done":
                            break

                    except websockets.exceptions.ConnectionClosed:
                        break
        except Exception as exc:
            logger.debug("openai_realtime_error", exc_info=True)
            return finalize_tts_result(
                provider="openai",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="openai",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
