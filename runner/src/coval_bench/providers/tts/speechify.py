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
from coval_bench.providers.tts._common import Synthesis

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000

_BASE_URL = "https://api.speechify.ai"
_OUTPUT_FORMAT = "pcm_24000"


class SpeechifyTTSProvider(TTSProvider):
    """Speechify TTS provider over the REST streaming endpoint."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
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

        synthesis = Synthesis("speechify", self._model, self._voice, SAMPLE_RATE)

        try:
            synthesis.start = time.monotonic()
            async with client.stream(
                "POST", "/v1/audio/stream", headers=headers, json=payload
            ) as response:
                synthesis.http_version = response.http_version
                synthesis.submit_to_headers_ms = submit_to_headers_ms(response.request)
                synthesis.connection_reused = connection_reused(response.request)
                if response.is_error:
                    body = await response.aread()
                    detail = body.decode("utf-8", "replace").strip() or response.reason_phrase
                    synthesis.start = None
                    synthesis.error = f"HTTP {response.status_code}: {detail[:500]}"
                    synthesis.status_code = response.status_code
                    return synthesis.result()
                async for chunk in response.aiter_bytes():
                    synthesis.add_chunk(chunk)
        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
