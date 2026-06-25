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
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

MODEL_ID = "canopylabs/orpheus-v1-english"

VALID_VOICES = ["autumn", "diana", "hannah", "austin", "daniel", "troy"]
_DEFAULT_VOICE = "autumn"

_MAX_INPUT_CHARS = 200

_API_BASE_URL = "https://api.groq.com/openai/v1"
_HOST_BASE_URL = "https://api.groq.com"
_FALLBACK_SAMPLE_RATE = 24000


class GroqTTSProvider(TTSProvider):
    _VALID_MODELS = frozenset({MODEL_ID})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in self._VALID_MODELS:
            raise ValueError(
                f"Invalid Groq TTS model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
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
        if len(text) > _MAX_INPUT_CHARS:
            return finalize_tts_result(
                provider="groq",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_FALLBACK_SAMPLE_RATE,
                audio_synthesis_start=None,
                first_audio_chunk_at=None,
                error=(
                    f"input is {len(text)} chars; Orpheus caps input at "
                    f"{_MAX_INPUT_CHARS} and would truncate the audio"
                ),
            )

        audio_chunks: list[bytes] = []
        http_version: str | None = None
        setup_ms: float | None = None
        reused: bool | None = None
        start: float | None = None
        first_chunk_at: float | None = None

        try:
            start = time.monotonic()
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=self._voice,
                input=text,
                response_format="wav",
            ) as response:
                http_version = response.http_version
                setup_ms = submit_to_headers_ms(response.http_response.request)
                reused = connection_reused(response.http_response.request)
                async for chunk in response.iter_bytes():
                    if isinstance(chunk, bytes) and len(chunk) > 0:
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        audio_chunks.append(chunk)
        except Exception as exc:
            logger.warning("groq_http_error", provider="groq", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="groq",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_FALLBACK_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
                http_version=http_version,
                submit_to_headers_ms=setup_ms,
                connection_reused=reused,
            )

        pcm, sample_rate, decode_error = _wav_to_pcm(b"".join(audio_chunks))
        return finalize_tts_result(
            provider="groq",
            model=self._model,
            voice=self._voice,
            pcm=pcm,
            sample_rate=sample_rate,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
            error=decode_error,
            http_version=http_version,
            submit_to_headers_ms=setup_ms,
            connection_reused=reused,
        )


def _wav_to_pcm(data: bytes) -> tuple[bytes, int, str | None]:
    if not data:
        return b"", _FALLBACK_SAMPLE_RATE, "no audio bytes received from Groq"
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_channels = wf.getnchannels()
            pcm = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        return b"", _FALLBACK_SAMPLE_RATE, f"Groq returned non-WAV audio: {exc}"
    if n_channels != 1 or sampwidth != 2:
        return (
            b"",
            sample_rate,
            f"expected mono 16-bit WAV from Groq; got {n_channels}ch/{sampwidth * 8}-bit",
        )
    return pcm, sample_rate, None
