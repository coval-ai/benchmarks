# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Airy TTS over HTTP streaming: 24 kHz mono signed 16-bit little-endian PCM.

The stream endpoint emits raw PCM; ``finalize_tts_result`` wraps it in a
24 kHz WAV and adds leading silence to the first-chunk arrival time.
Protocol: https://airy.so/cloud-api/docs/api/tts/speech-synthesis-stream
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

_BASE_URL = "https://api.airy.so"
_STREAM_PATH = "/v1/audio/speech/stream"


class AiryTTSProvider(TTSProvider):
    """Synthesize English benchmark prompts with airy-tts-v1 in normal style."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        api_key_secret = settings.airy_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value():
            raise ValueError("airy_api_key is required in Settings")

        self._model = model
        self._voice = voice
        self._api_key = api_key_secret.get_secret_value()
        # HTTP header encoding errors can include the entire Authorization value.
        if any(not 33 <= ord(char) <= 126 for char in self._api_key):
            raise ValueError("airy_api_key must contain only visible ASCII without whitespace")
        self._client = get_shared_client("airy", _BASE_URL)

    @property
    def name(self) -> str:
        return f"airy-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Warm the HTTP connection with HEAD, without submitting synthesis.

        Even a 401/405 response establishes a connection. The shared pool and
        per-request diagnostics follow the other HTTP TTS providers.
        """
        client = get_shared_client("airy", _BASE_URL)
        start = time.monotonic()
        response = await client.head(_STREAM_PATH)
        logger.info(
            "airy_prewarm",
            warmup_ms=round((time.monotonic() - start) * 1000, 1),
            http_version=response.http_version,
        )
        if response.http_version != "HTTP/2":
            logger.warning("airy_prewarm_no_http2", http_version=response.http_version)

    async def synthesize(self, text: str) -> TTSResult:
        synthesis = Synthesis("airy", self._model, self._voice, SAMPLE_RATE)

        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "language": "en",
            "style": "normal",
        }
        synthesis.start = time.monotonic()
        try:
            async with self._client.stream(
                "POST",
                _STREAM_PATH,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            ) as response:
                synthesis.status_code = response.status_code
                synthesis.http_version = response.http_version
                synthesis.submit_to_headers_ms = submit_to_headers_ms(response.request)
                synthesis.connection_reused = connection_reused(response.request)
                response.raise_for_status()
                _validate_audio_headers(response.headers)
                # No chunk_size: buffering to a fixed byte count would inflate TTFA.
                async for chunk in response.aiter_bytes():
                    synthesis.add_chunk(chunk)
        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()


def _validate_audio_headers(headers: httpx.Headers) -> None:
    content_type = headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type != "audio/pcm":
        raise ValueError(f"Expected Airy audio/pcm response, got {content_type!r}")

    # These are the model's pinned defaults. Reject a declared mismatch instead
    # of resampling or silently writing PCM with the wrong WAV metadata.
    expected = {
        "X-Audio-Sample-Rate": str(SAMPLE_RATE),
        "X-Audio-Channels": "1",
        "X-Audio-Sample-Format": "s16le",
    }
    for name, value in expected.items():
        actual = headers.get(name, value)
        if actual != value:
            raise ValueError(f"Expected Airy {name}={value}, got {actual!r}")
