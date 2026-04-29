# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""ElevenLabs TTS provider — uses the elevenlabs SDK for streaming synthesis."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import wave
from io import BytesIO
from pathlib import Path

import structlog
from elevenlabs import ElevenLabs

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000

# Models enabled for benchmarking
SUPPORTED_MODELS = {
    "eleven_flash_v2_5",
    "eleven_multilingual_v2",
    "eleven_turbo_v2_5",
}


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs TTS provider using the official SDK streaming endpoint."""

    enabled: bool = True

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

        api_key_secret = settings.elevenlabs_api_key
        if api_key_secret is None:
            raise ValueError("elevenlabs_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"elevenlabs-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _is_audio_chunk(message: object) -> bool:
        """Return True when *message* carries audio data."""
        if isinstance(message, bytes) and len(message) > 0:
            return True
        if isinstance(message, str):
            try:
                data = json.loads(message)
                return bool(data.get("audio"))
            except (json.JSONDecodeError, TypeError):
                return False
        return False

    @staticmethod
    def _extract_audio(message: object) -> bytes:
        """Extract raw PCM bytes from a message."""
        if isinstance(message, bytes):
            return message
        if isinstance(message, str):
            try:
                data = json.loads(message)
                audio_b64 = data.get("audio")
                if audio_b64:
                    return base64.b64decode(audio_b64)
            except (json.JSONDecodeError, TypeError, Exception):  # noqa: BLE001
                logger.debug("elevenlabs_extract_audio_failed", message=message[:100])
        return b""

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via ElevenLabs SDK and return a TTSResult."""
        ttfa_ms: float | None = None
        audio_stream = BytesIO()

        try:
            client = ElevenLabs(api_key=self._api_key)
            start = time.monotonic()

            response = client.text_to_speech.convert(
                voice_id=self._voice,
                output_format="pcm_24000",
                text=text,
                model_id=self._model,
            )

            for chunk in response:
                if chunk:
                    if self._is_audio_chunk(chunk) and ttfa_ms is None:
                        ttfa_ms = (time.monotonic() - start) * 1000
                        logger.debug(
                            "elevenlabs_ttfa",
                            model=self._model,
                            ttfa_ms=ttfa_ms,
                        )
                    if isinstance(chunk, bytes):
                        audio_stream.write(chunk)
                    else:
                        audio_stream.write(self._extract_audio(chunk))

        except Exception as exc:
            logger.debug("elevenlabs_error", exc_info=True)
            return TTSResult(
                provider="elevenlabs",
                model=self._model,
                voice=self._voice,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_data = audio_stream.getvalue()
        audio_path: Path | None = None
        if audio_data:
            audio_path = _write_wav_from_data(audio_data, SAMPLE_RATE)
        else:
            logger.warning("elevenlabs_no_audio", model=self._model)

        return TTSResult(
            provider="elevenlabs",
            model=self._model,
            voice=self._voice,
            ttfa_ms=ttfa_ms,
            audio_path=audio_path,
            error=None,
        )


def _write_wav_from_data(audio_data: bytes, sample_rate: int) -> Path:
    """Write raw PCM data as a WAV file and return the path."""
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(tmp_name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return Path(tmp_name)
