# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Speechify TTS provider — chunked HTTP streaming against a shared httpx pool."""

from __future__ import annotations

import time

import structlog

from coval_bench.config import Settings
from coval_bench.providers._http_session import (
    connection_reused,
    get_shared_client,
    submit_to_headers_ms,
)
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000

_BASE_URL = "https://api.speechify.ai"
_OUTPUT_FORMAT = "pcm_24000"


class SpeechifyTTSProvider(TTSProvider):
    """Speechify TTS provider over the REST streaming endpoint."""

    _VALID_MODELS = frozenset({"simba-3.2", "simba-3.0"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in self._VALID_MODELS:
            raise ValueError(
                f"Unsupported Speechify model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._model = model
        self._voice = voice

        api_key_secret = settings.speechify_api_key
        if api_key_secret is None:
            raise ValueError("speechify_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"speechify-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Pre-warm the shared httpx pool; a 401 on the HEAD still warms the socket."""
        client = get_shared_client("speechify", _BASE_URL)
        t0 = time.monotonic()
        response = await client.head("/v1/voices")
        logger.info(
            "speechify_prewarm",
            warmup_ms=round((time.monotonic() - t0) * 1000, 1),
            http_version=response.http_version,
        )
        if response.http_version != "HTTP/2":
            logger.warning("speechify_prewarm_no_http2", http_version=response.http_version)

    async def synthesize(self, text: str) -> TTSResult:
        client = get_shared_client("speechify", _BASE_URL)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        payload = {
            "input": text,
            "voice_id": self._voice,
            "model": self._model,
            "output_format": _OUTPUT_FORMAT,
        }

        audio_chunks: list[bytes] = []
        http_version: str | None = None
        setup_ms: float | None = None
        reused: bool | None = None
        start: float | None = None
        first_chunk_at: float | None = None

        try:
            start = time.monotonic()
            async with client.stream(
                "POST", "/v1/audio/stream", headers=headers, json=payload
            ) as response:
                http_version = response.http_version
                setup_ms = submit_to_headers_ms(response.request)
                reused = connection_reused(response.request)
                if response.is_error:
                    body = await response.aread()
                    detail = body.decode("utf-8", "replace").strip() or response.reason_phrase
                    return finalize_tts_result(
                        provider="speechify",
                        model=self._model,
                        voice=self._voice,
                        pcm=b"",
                        sample_rate=SAMPLE_RATE,
                        audio_synthesis_start=None,
                        first_audio_chunk_at=None,
                        error=f"HTTP {response.status_code}: {detail[:500]}",
                        http_version=http_version,
                        submit_to_headers_ms=setup_ms,
                        connection_reused=reused,
                    )
                async for chunk in response.aiter_bytes():
                    if chunk:
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(chunk)
        except Exception as exc:
            logger.warning("speechify_error", provider="speechify", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="speechify",
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

        audio_data = b"".join(audio_chunks)
        if not audio_data:
            logger.warning("speechify_no_audio", model=self._model)

        return finalize_tts_result(
            provider="speechify",
            model=self._model,
            voice=self._voice,
            pcm=audio_data,
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
            http_version=http_version,
            submit_to_headers_ms=setup_ms,
            connection_reused=reused,
        )
