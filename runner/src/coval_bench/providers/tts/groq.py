# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Groq TTS provider (Orpheus) over the OpenAI-compatible HTTP streaming speech API.

Orpheus only emits a ``wav`` container (never raw ``pcm``), so streamed bytes are
de-containered to PCM before scoring, and ``input`` is capped at 200 characters
server-side.
"""

from __future__ import annotations

import io
import time
import wave

import structlog
from openai import AsyncOpenAI

from coval_bench.config import Settings
from coval_bench.providers._http_session import (
    connection_reused,
    get_shared_client,
    submit_to_headers_ms,
)
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "canopylabs/orpheus-v1-english"

VALID_VOICES = ["autumn", "diana", "hannah", "austin", "daniel", "troy"]
_DEFAULT_VOICE = "autumn"

_MAX_INPUT_CHARS = 200

_API_BASE_URL = "https://api.groq.com/openai/v1"
_HOST_BASE_URL = "https://api.groq.com"
_SAMPLE_RATE = 24000


class GroqTTSProvider(TTSProvider):
    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model

        self._voice = voice
        if self._voice not in VALID_VOICES:
            logger.warning("unknown_groq_voice", voice=self._voice, fallback=_DEFAULT_VOICE)
            self._voice = _DEFAULT_VOICE

        api_key_secret = settings.groq_api_key
        if api_key_secret is None:
            raise ValueError("groq_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=_API_BASE_URL,
            http_client=get_shared_client("groq", _HOST_BASE_URL),
        )

    @property
    def name(self) -> str:
        return f"groq-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        client = get_shared_client("groq", _HOST_BASE_URL)
        t0 = time.monotonic()
        response = await client.head("/openai/v1/models")
        logger.info(
            "groq_prewarm",
            warmup_ms=round((time.monotonic() - t0) * 1000, 1),
            http_version=response.http_version,
        )
        if response.http_version != "HTTP/2":
            logger.warning("groq_prewarm_no_http2", http_version=response.http_version)

    async def synthesize(self, text: str) -> TTSResult:
        synthesis = Synthesis("groq", self._model, self._voice, _SAMPLE_RATE)
        if len(text) > _MAX_INPUT_CHARS:
            synthesis.error = (
                f"input is {len(text)} chars; Orpheus caps input at "
                f"{_MAX_INPUT_CHARS} and would truncate the audio"
            )
            return synthesis.result()

        try:
            synthesis.start = time.monotonic()
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=self._voice,
                input=text,
                response_format="wav",
                # Groq extension; the WAV header stays authoritative at decode.
                extra_body={"sample_rate": _SAMPLE_RATE},
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

        pcm, synthesis.sample_rate, synthesis.error = _wav_to_pcm(b"".join(synthesis.chunks))
        synthesis.chunks = [pcm]
        return synthesis.result()


def _wav_to_pcm(data: bytes) -> tuple[bytes, int, str | None]:
    if not data:
        return b"", _SAMPLE_RATE, "no audio bytes received from Groq"
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_channels = wf.getnchannels()
            pcm = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        return b"", _SAMPLE_RATE, f"Groq returned non-WAV audio: {exc}"
    if n_channels != 1 or sampwidth != 2:
        return (
            b"",
            sample_rate,
            f"expected mono 16-bit WAV from Groq; got {n_channels}ch/{sampwidth * 8}-bit",
        )
    return pcm, sample_rate, None
