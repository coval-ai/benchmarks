# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rime TTS provider — HTTP POST streaming to Rime TTS API."""

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
RIME_TTS_URL = "https://users.rime.ai/v1/rime-tts"

VALID_MODELS = {"arcana", "mistv2", "mistv3"}


class RimeTTSProvider(TTSProvider):
    """Rime TTS provider using HTTP streaming."""

    enabled: bool = False  # Commented out in legacy run_tts.py

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

        api_key_secret = settings.rime_api_key
        if api_key_secret is None:
            raise ValueError("rime_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"rime-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _is_audio_chunk(chunk: object) -> bool:
        return isinstance(chunk, bytes) and len(chunk) > 0

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via Rime HTTP and return a TTSResult."""
        if self._model not in VALID_MODELS:
            return TTSResult(
                provider="rime",
                model=self._model,
                voice=self._voice,
                ttfa_ms=None,
                audio_path=None,
                error=(
                    f"Unsupported Rime model: {self._model}. Valid models: {sorted(VALID_MODELS)}"
                ),
            )

        audio_chunks: list[bytes] = []
        ttfa_ms: float | None = None

        payload = {
            "speaker": self._voice or "luna",
            "text": text,
            "modelId": self._model,
            "repetition_penalty": 1.5,
            "temperature": 0.5,
            "top_p": 1,
            "samplingRate": SAMPLE_RATE,
            "max_tokens": 1200,
        }

        headers = {
            "Accept": "audio/pcm",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                start = time.monotonic()
                async with session.post(
                    RIME_TTS_URL,
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Rime HTTP error {response.status}: {error_text}")

                    async for chunk in response.content.iter_any():
                        if self._is_audio_chunk(chunk):
                            if ttfa_ms is None:
                                ttfa_ms = (time.monotonic() - start) * 1000
                                logger.debug(
                                    "rime_ttfa",
                                    model=self._model,
                                    ttfa_ms=ttfa_ms,
                                )
                            audio_chunks.append(chunk)

        except Exception as exc:
            logger.debug("rime_error", exc_info=True)
            return TTSResult(
                provider="rime",
                model=self._model,
                voice=self._voice,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_path = _write_wav(audio_chunks, SAMPLE_RATE) if audio_chunks else None
        return TTSResult(
            provider="rime",
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
