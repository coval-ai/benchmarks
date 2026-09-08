# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Guava TTS provider (daytona-tts).

Wire protocol: HTTP POST, <guava_base_url>/audio/speech, streaming response.
Auth: Authorization: Bearer <key>.
Request (JSON): {"input": text, "voice": ..., "sampling_rate": 16000,
  "response_format": "wav", "stream": true}
Response: chunked 16 kHz mono PCM bytes; TTFA = submit -> first chunk.
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

SAMPLE_RATE = 16000

_ENDPOINT = "/audio/speech"


class GuavaTTSProvider(TTSProvider):
    """Guava TTS provider."""

    _VALID_MODELS = frozenset({"daytona-tts"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in self._VALID_MODELS:
            raise ValueError(
                f"Unsupported Guava model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._model = model
        self._voice = voice

        if not settings.guava_base_url:
            raise ValueError("guava_base_url is required in Settings")
        self._base_url = settings.guava_base_url

        api_key_secret = settings.guava_api_key
        if api_key_secret is None:
            raise ValueError("guava_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"guava-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Wake and warm the self-hosted endpoint; deletes its artifact; never fatal."""
        if not settings.guava_base_url or settings.guava_api_key is None:
            return
        provider = cls(settings, "daytona-tts", "grace")
        t0 = time.monotonic()
        result = await provider.synthesize("Warm up.")
        if result.audio_path is not None:
            result.audio_path.unlink(missing_ok=True)
        logger.info(
            "guava_prewarm",
            provider="guava",
            model="daytona-tts",
            warmup_ms=round((time.monotonic() - t0) * 1000, 1),
            error=result.error,
        )

    async def synthesize(self, text: str) -> TTSResult:
        client = get_shared_client("guava", self._base_url)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": text,
            "voice": self._voice,
            "sampling_rate": SAMPLE_RATE,
            "response_format": "wav",
            "stream": True,
        }

        audio_chunks: list[bytes] = []
        http_version: str | None = None
        setup_ms: float | None = None
        reused: bool | None = None
        start: float | None = None
        first_chunk_at: float | None = None

        try:
            start = time.monotonic()
            async with client.stream("POST", _ENDPOINT, headers=headers, json=payload) as response:
                http_version = response.http_version
                setup_ms = submit_to_headers_ms(response.request)
                reused = connection_reused(response.request)
                if response.is_error:
                    body = await response.aread()
                    detail = body.decode("utf-8", "replace").strip() or response.reason_phrase
                    return finalize_tts_result(
                        provider="guava",
                        model=self._model,
                        voice=self._voice,
                        pcm=b"",
                        sample_rate=SAMPLE_RATE,
                        audio_synthesis_start=None,
                        first_audio_chunk_at=None,
                        error=f"HTTP {response.status_code}: {detail[:500]}",
                        status_code=response.status_code,
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
            logger.warning("guava_error", provider="guava", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="guava",
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
            logger.warning("guava_no_audio", model=self._model)

        return finalize_tts_result(
            provider="guava",
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
