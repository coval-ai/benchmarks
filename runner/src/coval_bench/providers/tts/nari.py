# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Nari Labs TTS over HTTP streaming: 24 kHz mono signed 16-bit little-endian PCM.

Nari serves Qwen3-TTS behind an OpenAI-shaped ``/v1/audio/speech`` endpoint.
Protocol: https://docs.narilabs.com/generate-speech
"""

from __future__ import annotations

import time

import httpx
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

_BASE_URL = "https://api.narilabs.com"
_SPEECH_PATH = "/v1/audio/speech"


class NariTTSProvider(TTSProvider):
    """Synthesize English benchmark prompts with Nari's hosted Qwen3-TTS models."""

    _VALID_MODELS = frozenset({"qwen3-tts", "qwen3-tts-fast"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Nari TTS model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        api_key_secret = settings.nari_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value():
            raise ValueError("nari_api_key is required in Settings")

        self._model = model
        self._voice = voice
        self._api_key = api_key_secret.get_secret_value()
        # HTTP header encoding errors can include the entire Authorization value.
        if any(not 33 <= ord(char) <= 126 for char in self._api_key):
            raise ValueError("nari_api_key must contain only visible ASCII without whitespace")
        self._client = get_shared_client("nari", _BASE_URL)

    @property
    def name(self) -> str:
        return f"nari-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Warm the HTTP connection with HEAD, without submitting synthesis."""
        client = get_shared_client("nari", _BASE_URL)
        start = time.monotonic()
        response = await client.head(_SPEECH_PATH)
        logger.info(
            "nari_prewarm",
            warmup_ms=round((time.monotonic() - start) * 1000, 1),
            http_version=response.http_version,
        )
        if response.http_version != "HTTP/2":
            logger.warning("nari_prewarm_no_http2", http_version=response.http_version)

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        first_chunk_at: float | None = None
        status_code: int | None = None
        http_version: str | None = None
        setup_ms: float | None = None
        reused: bool | None = None
        request_id: str | None = None
        error: str | None = None

        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "language": "en",
            "stream": True,
            "response_format": "pcm",
        }
        start = time.monotonic()
        try:
            async with self._client.stream(
                "POST",
                _SPEECH_PATH,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            ) as response:
                status_code = response.status_code
                http_version = response.http_version
                setup_ms = submit_to_headers_ms(response.request)
                reused = connection_reused(response.request)
                request_id = response.headers.get("x-request-id")
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {body[:400]}",
                        request=response.request,
                        response=response,
                    )
                _validate_content_type(response.headers)
                # No chunk_size: buffering to a fixed byte count would inflate TTFA.
                async for chunk in response.aiter_bytes():
                    if chunk:
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(chunk)
        except Exception as exc:
            logger.warning(
                "nari_tts_error",
                provider="nari",
                model=self._model,
                request_id=request_id,
                exc_info=exc,
            )
            error = str(exc) or type(exc).__name__
            if request_id:
                error = f"{error} (x-request-id={request_id})"
            # A partial stream must not be saved/scored as complete synthesis.
            audio_chunks.clear()

        return finalize_tts_result(
            provider="nari",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
            error=error,
            status_code=status_code,
            http_version=http_version,
            submit_to_headers_ms=setup_ms,
            connection_reused=reused,
        )


def _validate_content_type(headers: httpx.Headers) -> None:
    content_type = headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type != "audio/pcm":
        raise ValueError(f"Expected Nari audio/pcm response, got {content_type!r}")
