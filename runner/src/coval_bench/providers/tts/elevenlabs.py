# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""ElevenLabs TTS provider — direct REST against a shared httpx pool.

TTFA is measured from the first request byte to the first PCM chunk. The
shared client is pre-warmed by ``warmup`` before the dataset loop, so the
measurement excludes TCP+TLS setup.
"""

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

_BASE_URL = "https://api.elevenlabs.io"
_OUTPUT_FORMAT = "pcm_24000"


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs TTS provider over the REST streaming endpoint."""

    _VALID_MODELS = frozenset({"eleven_flash_v2_5", "eleven_multilingual_v2", "eleven_turbo_v2_5"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in self._VALID_MODELS:
            raise ValueError(
                f"Unsupported ElevenLabs model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._model = model
        self._voice = voice

        api_key_secret = settings.elevenlabs_api_key
        if api_key_secret is None:
            raise ValueError("elevenlabs_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"elevenlabs-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Pre-warm the shared httpx connection pool with a HEAD probe.

        Transport failures propagate to the caller, which runs warmup under
        ``return_exceptions=True`` and logs them. A 401 still warms the
        socket, so an unauthenticated HEAD is sufficient.
        """
        client = get_shared_client("elevenlabs", _BASE_URL)
        t0 = time.monotonic()
        response = await client.head("/v1/voices")
        logger.info(
            "elevenlabs_prewarm",
            warmup_ms=round((time.monotonic() - t0) * 1000, 1),
            http_version=response.http_version,
        )
        if response.http_version != "HTTP/2":
            logger.warning("elevenlabs_prewarm_no_http2", http_version=response.http_version)

    async def synthesize(self, text: str) -> TTSResult:
        client = get_shared_client("elevenlabs", _BASE_URL)
        url = f"/v1/text-to-speech/{self._voice}/stream?output_format={_OUTPUT_FORMAT}"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        payload = {"text": text, "model_id": self._model}

        audio_chunks: list[bytes] = []
        http_version: str | None = None
        setup_ms: float | None = None
        reused: bool | None = None
        start: float | None = None
        first_chunk_at: float | None = None

        try:
            start = time.monotonic()
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                http_version = response.http_version
                setup_ms = submit_to_headers_ms(response.request)
                reused = connection_reused(response.request)
                if response.is_error:
                    body = await response.aread()
                    detail = body.decode("utf-8", "replace").strip() or response.reason_phrase
                    return finalize_tts_result(
                        provider="elevenlabs",
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
                    if chunk and first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                    if chunk:
                        audio_chunks.append(chunk)
        except Exception as exc:
            logger.debug("elevenlabs_error", exc_info=True)
            return finalize_tts_result(
                provider="elevenlabs",
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
            logger.warning("elevenlabs_no_audio", model=self._model)

        return finalize_tts_result(
            provider="elevenlabs",
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
