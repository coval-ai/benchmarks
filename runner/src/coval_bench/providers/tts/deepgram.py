# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deepgram TTS provider — HTTP POST streaming to Deepgram Speak API."""

from __future__ import annotations

import os
import tempfile
import time
import wave
from pathlib import Path

import aiohttp
import structlog

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"


class DeepgramTTSProvider(TTSProvider):
    """Deepgram TTS provider using HTTP streaming (Speak API)."""

    enabled: bool = True

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

        api_key_secret = settings.deepgram_api_key
        if api_key_secret is None:
            raise ValueError("deepgram_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"deepgram-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _is_audio_chunk(chunk: object) -> bool:
        if isinstance(chunk, bytes) and len(chunk) > 0:
            return True
        return bool(hasattr(chunk, "audio") and getattr(chunk, "audio", None)) or bool(
            hasattr(chunk, "data") and getattr(chunk, "data", None)
        )

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via Deepgram HTTP and return a TTSResult."""
        audio_chunks: list[bytes] = []
        ttfa_ms: float | None = None

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json",
        }
        params = {
            "model": self._model,
            "encoding": "linear16",
            "sample_rate": str(SAMPLE_RATE),
        }
        payload = {"text": text}

        try:
            async with aiohttp.ClientSession() as session:
                start = time.monotonic()
                async with session.post(
                    DEEPGRAM_TTS_URL,
                    headers=headers,
                    params=params,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Deepgram HTTP error {response.status}: {error_text}")

                    async for chunk in response.content.iter_any():
                        if self._is_audio_chunk(chunk):
                            if ttfa_ms is None:
                                ttfa_ms = (time.monotonic() - start) * 1000
                                logger.debug(
                                    "deepgram_ttfa",
                                    model=self._model,
                                    ttfa_ms=ttfa_ms,
                                )
                            audio_chunks.append(chunk)

        except Exception as exc:
            logger.debug("deepgram_error", exc_info=True)
            return TTSResult(
                provider="deepgram",
                model=self._model,
                voice=self._voice,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_path = _write_wav(audio_chunks, SAMPLE_RATE) if audio_chunks else None
        return TTSResult(
            provider="deepgram",
            model=self._model,
            voice=self._voice,
            ttfa_ms=ttfa_ms,
            audio_path=audio_path,
            error=None,
        )


def _write_wav(chunks: list[bytes], sample_rate: int) -> Path:
    """Concatenate PCM chunks and write a WAV file to a temp location."""
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio_data = b"".join(chunks)
    with wave.open(tmp_name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return Path(tmp_name)
