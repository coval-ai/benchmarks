# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deepgram Aura TTS hosted on Cloudflare Workers AI: 24 kHz mono PCM16 over HTTP.

POST /client/v4/accounts/<account>/ai/run/@cf/deepgram/<model> with a JSON body
returns the audio bytes directly. Errors come back as JSON ``{"errors": [...]}``.
"""

from __future__ import annotations

import json
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

_BASE_URL = "https://api.cloudflare.com"


def _run_path(account_id: str, model: str) -> str:
    return f"/client/v4/accounts/{account_id}/ai/run/@cf/deepgram/{model}"


class CloudflareTTSProvider(TTSProvider):
    """Deepgram Aura TTS served from Cloudflare Workers AI."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        api_key_secret = settings.cloudflare_api_key
        if api_key_secret is None or not api_key_secret.get_secret_value():
            raise ValueError("cloudflare_api_key is required in Settings")
        if not settings.cloudflare_account_id:
            raise ValueError("cloudflare_account_id is required in Settings")

        self._model = model
        self._voice = voice
        self._api_key = api_key_secret.get_secret_value()
        self._path = _run_path(settings.cloudflare_account_id, model)
        self._client = get_shared_client("cloudflare", _BASE_URL)

    @property
    def name(self) -> str:
        return f"cloudflare-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Warm the HTTP connection with HEAD, without submitting synthesis."""
        client = get_shared_client("cloudflare", _BASE_URL)
        start = time.monotonic()
        response = await client.head("/client/v4/accounts")
        logger.info(
            "cloudflare_prewarm",
            warmup_ms=round((time.monotonic() - start) * 1000, 1),
            http_version=response.http_version,
        )
        if response.http_version != "HTTP/2":
            logger.warning("cloudflare_prewarm_no_http2", http_version=response.http_version)

    async def synthesize(self, text: str) -> TTSResult:
        synthesis = Synthesis("cloudflare", self._model, self._voice, SAMPLE_RATE)
        request_id: str | None = None

        payload = {
            "text": text,
            "speaker": self._voice,
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "container": "none",
        }
        synthesis.start = time.monotonic()
        try:
            async with self._client.stream(
                "POST",
                self._path,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            ) as response:
                synthesis.status_code = response.status_code
                synthesis.http_version = response.http_version
                synthesis.submit_to_headers_ms = submit_to_headers_ms(response.request)
                synthesis.connection_reused = connection_reused(response.request)
                request_id = response.headers.get("cf-ai-req-id")
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {_error_message(body)}",
                        request=response.request,
                        response=response,
                    )
                # Workers AI labels the linear16 body audio/mpeg, so the content
                # type is not checked.
                async for chunk in response.aiter_bytes():
                    synthesis.add_chunk(chunk)
        except Exception as exc:
            synthesis.fail(exc, request_id=request_id)
            if request_id:
                synthesis.error = f"{synthesis.error} (cf-ai-req-id={request_id})"

        return synthesis.result()


def _error_message(body: str) -> str:
    try:
        errors = json.loads(body).get("errors")
        if isinstance(errors, list) and errors:
            return str(errors[0].get("message", body[:400]))
    except (ValueError, AttributeError):
        pass
    return body[:400]
