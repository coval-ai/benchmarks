# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseten dedicated-endpoint TTS provider (Qwen3-TTS over WebSocket).

Protocol: send a JSON ``session.config``, then ``input.text`` and ``input.done``;
the server streams back 24 kHz mono PCM16 as binary frames interleaved with JSON
status messages, finishing with ``session.done``. The endpoint URL embeds a
private model id, so it is read from settings rather than hardcoded.
"""

from __future__ import annotations

import asyncio
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
# Our warmup request is the scale-up trigger for a demand-scaled deployment;
# the budget must outlast a full boot (~166 s measured) with provisioning slack.
_WARMUP_TIMEOUT_S = 360


class BasetenTTSProvider(TTSProvider):
    """Baseten TTS provider using WebSocket streaming (Qwen3-TTS)."""

    def __init__(
        self,
        settings: Settings,
        model: str,
        voice: str,
        open_timeout_s: float = _OPEN_TIMEOUT_S,
    ) -> None:
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
        self._open_timeout_s = open_timeout_s

    @property
    def name(self) -> str:
        return f"baseten-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Synthesize a throwaway phrase so the single pinned replica is hot.

        Mirrors the Baseten STT warmup: Baseten is retiring its own scheduled
        warmup traffic and asked us to warm endpoints before testing. The audio
        artifact is deleted here — the orchestrator only cleans up items it ran.
        """
        api_key = settings.baseten_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            return
        if not settings.baseten_qwen_url:
            return
        provider = cls(
            settings, _VALID_MODELS[0], _VALID_VOICES[0], open_timeout_s=_WARMUP_TIMEOUT_S
        )
        t0 = time.monotonic()
        error: str | None = None
        try:
            async with asyncio.timeout(_WARMUP_TIMEOUT_S):
                result = await provider.synthesize("Warm up.")
            if result.audio_path is not None:
                result.audio_path.unlink(missing_ok=True)
            error = result.error
        except TimeoutError:
            error = f"warmup timed out after {_WARMUP_TIMEOUT_S}s; endpoint still not up"
        logger.info(
            "baseten_tts_prewarm",
            provider="baseten",
            model=_VALID_MODELS[0],
            warmup_ms=round((time.monotonic() - t0) * 1000, 1),
            error=error,
        )

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
                open_timeout=self._open_timeout_s,
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
