# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""AssemblyAI TTS provider — OpenAI-compatible HTTP streaming synthesis.

Endpoint: POST https://tts.us.assemblyai.com/v1/openai/audio/speech
Voices and the output rate are set with the ``voice_id`` and ``sample_rate``
extension fields; OpenAI's ``voice`` parameter must be left out or the request
is refused with a 400.
"""

from __future__ import annotations

import re
import time

import structlog
from openai import NOT_GIVEN, AsyncOpenAI

from coval_bench.config import Settings
from coval_bench.providers._http_session import (
    connection_reused,
    get_shared_client,
    submit_to_headers_ms,
)
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Provider voice slugs. Cloned (Arena) voices are addressed by their UUID instead.
VALID_VOICES = (
    "aai-isaiah-2",
    "aai-eric-2",
    "aai-audrey",
    "aai-emilia-2",
    "aai-cameron-2",
    "aai-eric-6",
    "aai-nancy",
    "aai-carrie",
)
_VOICE_UUID = re.compile(r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$")
SAMPLE_RATE = 44100

_BASE_URL = "https://tts.us.assemblyai.com"
_OPENAI_PREFIX = "/v1/openai"


def _voice_supported(voice: str) -> bool:
    return voice in VALID_VOICES or _VOICE_UUID.match(voice) is not None


class AssemblyAITTSProvider(TTSProvider):
    """AssemblyAI TTS over the OpenAI-compatible streaming speech API."""

    _VALID_MODELS = frozenset({"assemblyai-tts-1"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid AssemblyAI TTS model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if not _voice_supported(voice):
            raise ValueError(
                f"Invalid AssemblyAI TTS voice {voice!r}. Valid: {VALID_VOICES} or a voice UUID"
            )
        self._model = model
        self._voice = voice

        api_key_secret = settings.assemblyai_api_key
        if api_key_secret is None:
            raise ValueError("assemblyai_api_key is required in Settings")
        self._client = AsyncOpenAI(
            api_key=api_key_secret.get_secret_value(),
            base_url=f"{_BASE_URL}{_OPENAI_PREFIX}",
            http_client=get_shared_client("assemblyai", _BASE_URL),
        )

    @property
    def name(self) -> str:
        return f"assemblyai-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Pre-warm the shared httpx pool. A 401 still warms the socket."""
        client = get_shared_client("assemblyai", _BASE_URL)
        t0 = time.monotonic()
        response = await client.head(f"{_OPENAI_PREFIX}/audio/speech")
        logger.info(
            "assemblyai_tts_prewarm",
            warmup_ms=round((time.monotonic() - t0) * 1000, 1),
            http_version=response.http_version,
        )

    async def synthesize(self, text: str) -> TTSResult:
        synthesis = Synthesis("assemblyai", self._model, self._voice, SAMPLE_RATE)

        try:
            synthesis.start = time.monotonic()
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                # The SDK types voice as required; NOT_GIVEN drops it from the body.
                voice=NOT_GIVEN,  # type: ignore[arg-type]
                input=text,
                response_format="pcm",
                extra_body={"voice_id": self._voice, "sample_rate": SAMPLE_RATE},
            ) as response:
                synthesis.http_version = response.http_version
                synthesis.submit_to_headers_ms = submit_to_headers_ms(
                    response.http_response.request
                )
                synthesis.connection_reused = connection_reused(response.http_response.request)
                async for chunk in response.iter_bytes():
                    if isinstance(chunk, bytes):
                        synthesis.add_chunk(chunk)
        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
