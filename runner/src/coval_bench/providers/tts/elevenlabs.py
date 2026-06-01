# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""ElevenLabs TTS provider — uses the elevenlabs SDK for streaming synthesis."""

from __future__ import annotations

import base64
import json
import time
from io import BytesIO

import structlog
from elevenlabs import ElevenLabs

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs TTS provider using the official SDK streaming endpoint."""

    enabled: bool = True

    _VALID_MODELS = frozenset({"eleven_flash_v2_5", "eleven_multilingual_v2", "eleven_turbo_v2_5"})

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
        if not self._model_supported(self._model):
            return TTSResult(
                provider="elevenlabs",
                model=self._model,
                voice=self._voice,
                ttfa_ms=None,
                audio_path=None,
                error=(
                    f"Unsupported ElevenLabs model: {self._model}. "
                    f"Valid models: {sorted(self._VALID_MODELS)}"
                ),
            )
        start: float | None = None
        first_chunk_at: float | None = None
        audio_stream = BytesIO()

        try:
            client = ElevenLabs(api_key=self._api_key)
            start = time.monotonic()

            response = client.text_to_speech.stream(
                voice_id=self._voice,
                output_format="pcm_24000",
                text=text,
                model_id=self._model,
            )

            for chunk in response:
                if chunk:
                    if self._is_audio_chunk(chunk) and first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                    if isinstance(chunk, bytes):
                        audio_stream.write(chunk)
                    else:
                        audio_stream.write(self._extract_audio(chunk))

        except Exception as exc:
            logger.debug("elevenlabs_error", exc_info=True)
            return finalize_tts_result(
                provider="elevenlabs",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        audio_data = audio_stream.getvalue()
        if not audio_data:
            logger.warning("elevenlabs_no_audio", model=self._model)

        return finalize_tts_result(
            provider="elevenlabs",
            model=self._model,
            voice=self._voice,
            pcm=audio_data,
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
