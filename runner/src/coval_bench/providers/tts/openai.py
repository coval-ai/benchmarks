# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI TTS provider — HTTP streaming synthesis."""

from __future__ import annotations

import time

import structlog
from openai import AsyncOpenAI

from coval_bench.config import Settings
from coval_bench.providers._http_session import (
    connection_reused,
    get_shared_client,
    submit_to_headers_ms,
)
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

HTTP_MODELS = {"gpt-4o-mini-tts"}
SAMPLE_RATE = 24000

_BASE_URL = "https://api.openai.com"


class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS provider over the HTTP streaming speech API."""

    _VALID_MODELS = frozenset(HTTP_MODELS)

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
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            http_client=get_shared_client("openai", _BASE_URL),
        )

    @property
    def name(self) -> str:
        return f"openai-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Pre-warm the shared httpx pool used by the HTTP TTS path.

        Transport failures propagate to the caller, which runs warmup under
        ``return_exceptions=True`` and logs them. A 401 still warms the
        socket, so an unauthenticated HEAD is sufficient.
        """
        client = get_shared_client("openai", _BASE_URL)
        t0 = time.monotonic()
        response = await client.head("/v1/models")
        logger.info(
            "openai_prewarm",
            warmup_ms=round((time.monotonic() - t0) * 1000, 1),
            http_version=response.http_version,
        )
        if response.http_version != "HTTP/2":
            logger.warning("openai_prewarm_no_http2", http_version=response.http_version)

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech and return a TTSResult."""
        if self._model in HTTP_MODELS:
            return await self._synthesize_http(text)
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
        http_version: str | None = None
        setup_ms: float | None = None
        reused: bool | None = None
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
                http_version = response.http_version
                setup_ms = submit_to_headers_ms(response.http_response.request)
                reused = connection_reused(response.http_response.request)
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
                http_version=http_version,
                submit_to_headers_ms=setup_ms,
                connection_reused=reused,
            )

        return finalize_tts_result(
            provider="openai",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
            http_version=http_version,
            submit_to_headers_ms=setup_ms,
            connection_reused=reused,
        )
