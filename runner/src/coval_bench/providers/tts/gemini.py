# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gemini TTS over the Interactions API (SSE streaming).

Distinct from the ``google`` provider, which reaches Gemini-TTS through Cloud
Text-to-Speech with service-account auth: the 3.8 generation is served only by
``POST /v1beta/interactions`` with API-key auth. Audio streams as base64
``step.delta`` events of headerless 16-bit little-endian PCM at 24 kHz.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

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

_BASE_URL = "https://generativelanguage.googleapis.com"
_INTERACTIONS_PATH = "/v1beta/interactions"
_MIME_TYPE = "audio/l16"


class GeminiTTSProvider(TTSProvider):
    """Synthesize with a prebuilt Gemini voice through the Interactions API."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        api_key_secret = settings.gemini_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value():
            raise ValueError("gemini_api_key is required in Settings")
        self._model = model
        self._voice = voice
        self._api_key = api_key_secret.get_secret_value()
        self._client = get_shared_client("gemini", _BASE_URL)

    @property
    def name(self) -> str:
        return f"gemini-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Warm the HTTP connection with HEAD, without submitting synthesis."""
        client = get_shared_client("gemini", _BASE_URL)
        start = time.monotonic()
        response = await client.head(_INTERACTIONS_PATH)
        logger.info(
            "gemini_tts_prewarm",
            warmup_ms=round((time.monotonic() - start) * 1000, 1),
            http_version=response.http_version,
        )
        if response.http_version != "HTTP/2":
            logger.warning("gemini_tts_prewarm_no_http2", http_version=response.http_version)

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        first_chunk_at: float | None = None
        status_code: int | None = None
        http_version: str | None = None
        setup_ms: float | None = None
        reused: bool | None = None
        error: str | None = None

        payload = {
            "model": self._model,
            "input": text,
            "response_format": {
                "type": "audio",
                "mime_type": _MIME_TYPE,
                "sample_rate": SAMPLE_RATE,
            },
            "generation_config": {"speech_config": [{"voice": self._voice}]},
            "stream": True,
        }
        start = time.monotonic()
        try:
            async with self._client.stream(
                "POST",
                _INTERACTIONS_PATH,
                headers={"x-goog-api-key": self._api_key},
                json=payload,
            ) as response:
                status_code = response.status_code
                http_version = response.http_version
                setup_ms = submit_to_headers_ms(response.request)
                reused = connection_reused(response.request)
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {_error_message(body)}",
                        request=response.request,
                        response=response,
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event: dict[str, Any] = json.loads(data)
                    event_type = event.get("event_type")
                    if event_type == "error":
                        raise RuntimeError(_error_message(json.dumps(event)))
                    if event_type != "step.delta":
                        continue
                    delta = event.get("delta") or {}
                    if delta.get("type") != "audio":
                        continue
                    chunk = base64.b64decode(delta.get("data") or "")
                    if not chunk:
                        continue
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                        _validate_audio_format(delta)
                    audio_chunks.append(chunk)
        except Exception as exc:
            logger.warning("gemini_tts_error", provider="gemini", model=self._model, exc_info=exc)
            error = str(exc) or type(exc).__name__
            audio_chunks.clear()

        return finalize_tts_result(
            provider="gemini",
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


def _validate_audio_format(delta: dict[str, Any]) -> None:
    # The requested format is pinned; a declared mismatch would be written as
    # PCM with the wrong WAV metadata rather than resampled.
    expected = {"mime_type": _MIME_TYPE, "sample_rate": SAMPLE_RATE, "channels": 1}
    for key, value in expected.items():
        actual = delta.get(key, value)
        if actual != value:
            raise ValueError(f"Expected Gemini {key}={value}, got {actual!r}")


def _error_message(body: str) -> str:
    # Interactions errors are {"error": {...}}; the shared Google API layer
    # (bad key, quota) wraps the same shape in a one-element list.
    try:
        parsed = json.loads(body)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        message = parsed.get("error", {}).get("message")
        if message:
            return str(message)
    except (ValueError, AttributeError):
        pass
    return body[:400]
