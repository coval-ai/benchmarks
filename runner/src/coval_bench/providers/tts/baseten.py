# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseten dedicated-endpoint TTS provider (Qwen3-TTS over WebSocket).

Protocol: send a JSON ``session.config``, then ``input.text`` and ``input.done``;
the server streams back 24 kHz mono PCM16 as binary frames interleaved with JSON
status messages, finishing with ``session.done``. The endpoint URL embeds a
private model id, so it is read from settings rather than hardcoded.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_VALID_MODELS = ("qwen3-tts-1.7b",)
_VALID_VOICES = ("lisa", "jim")
_SAMPLE_RATE = 24000
_MAX_WS_SIZE = 16 * 1024 * 1024
# Cold replicas exceed the 10 s websockets default for the handshake.
_OPEN_TIMEOUT_S = 45


class BasetenTTSProvider(TTSProvider):
    """Baseten TTS provider using WebSocket streaming (Qwen3-TTS)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Baseten TTS model {model!r}. Valid: {_VALID_MODELS}")
        if voice not in _VALID_VOICES:
            raise ValueError(f"Invalid Baseten TTS voice {voice!r}. Valid: {_VALID_VOICES}")
        self._model = model
        self._voice = voice

        api_key_secret = settings.baseten_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value().strip():
            raise ValueError("baseten_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

        if not settings.baseten_qwen_url:
            raise ValueError("baseten_qwen_url is required in Settings")
        self._ws_url = settings.baseten_qwen_url

    @property
    def name(self) -> str:
        return f"baseten-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None
        headers = {"Authorization": f"Api-Key {self._api_key}"}

        try:
            async with ws_client.connect(
                self._ws_url,
                additional_headers=headers,
                max_size=_MAX_WS_SIZE,
                open_timeout=_OPEN_TIMEOUT_S,
            ) as ws:
                # Clock starts post-handshake so TTFA excludes connect (cohort parity).
                start = time.monotonic()
                # x_vector_only_mode reuses the cached speaker embedding (fast path).
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.config",
                            "task_type": "Base",
                            "response_format": "pcm",
                            "stream_audio": True,
                            "speed": 1.0,
                            "split_granularity": "sentence",
                            "voice": self._voice,
                            "x_vector_only_mode": True,
                        }
                    )
                )
                await ws.send(json.dumps({"type": "input.text", "text": text}))
                await ws.send(json.dumps({"type": "input.done"}))

                async for message in ws:
                    if isinstance(message, (bytes, bytearray)):
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(bytes(message))
                        continue
                    data: dict[str, Any] = json.loads(message)
                    if data.get("type") == "session.done":
                        break
                    if data.get("type") == "error":
                        raise RuntimeError(str(data.get("message", data)))

        except Exception as exc:
            logger.warning("baseten_tts_error", provider="baseten", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="baseten",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="baseten",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
