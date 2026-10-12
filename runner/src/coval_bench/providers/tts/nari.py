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
from coval_bench.providers.tts._common import Synthesis

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000

_BASE_URL = "https://api.narilabs.com"
_SPEECH_PATH = "/v1/audio/speech"


class NariTTSProvider(TTSProvider):
    """Synthesize English benchmark prompts with Nari's hosted Qwen3-TTS models."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
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
        synthesis = Synthesis("nari", self._model, self._voice, SAMPLE_RATE)
        request_id: str | None = None

        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "language": "en",
            "stream": True,
            "response_format": "pcm",
        }
        synthesis.start = time.monotonic()
        try:
            async with self._client.stream(
                "POST",
                _SPEECH_PATH,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            ) as response:
                synthesis.status_code = response.status_code
                synthesis.http_version = response.http_version
                synthesis.submit_to_headers_ms = submit_to_headers_ms(response.request)
                synthesis.connection_reused = connection_reused(response.request)
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
                    synthesis.add_chunk(chunk)
        except Exception as exc:
            synthesis.fail(exc, request_id=request_id)
            if request_id:
                synthesis.error = f"{synthesis.error} (x-request-id={request_id})"

        return synthesis.result()


def _validate_content_type(headers: httpx.Headers) -> None:
    content_type = headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type != "audio/pcm":
        raise ValueError(f"Expected Nari audio/pcm response, got {content_type!r}")
