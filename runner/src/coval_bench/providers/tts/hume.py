# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hume TTS provider — requires the optional ``hume-tts`` extra.

Install with:
    pip install "coval-bench[hume-tts]"

If the ``hume`` package is not installed, importing this module raises
``ImportError``. The ``__init__.py`` registry catches this and sets
``HUME_AVAILABLE = False``. Tests should use
``pytest.importorskip("hume")`` to skip automatically when the extra
is absent.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import structlog

# This import is intentionally left to propagate ImportError if hume is absent.
from hume import HumeClient
from hume.tts import (
    FormatWav,
    PostedUtterance,
    PostedUtteranceVoiceWithId,
)

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000

# Models supported by this provider
SUPPORTED_MODELS = {"octave-tts", "octave-2"}

# Default voice ID for octave models (Hume AI built-in)
_DEFAULT_VOICE_ID = "176a55b1-4468-4736-8878-db82729667c1"


class HumeTTSProvider(TTSProvider):
    """Hume TTS provider (optional — requires ``hume-tts`` extra)."""

    enabled: bool = False  # Commented out in legacy run_tts.py

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

        api_key_secret = settings.hume_api_key
        if api_key_secret is None:
            raise ValueError("hume_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"hume-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _is_audio_chunk(chunk: object) -> bool:
        return isinstance(chunk, bytes) and len(chunk) > 0

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via Hume SDK and return a TTSResult."""
        if self._model not in SUPPORTED_MODELS:
            return TTSResult(
                provider="hume",
                model=self._model,
                voice=self._voice,
                ttfa_ms=None,
                audio_path=None,
                error=(
                    f"Unsupported Hume model: {self._model}. Supported: {sorted(SUPPORTED_MODELS)}"
                ),
            )

        audio_chunks: list[bytes] = []
        ttfa_ms: float | None = None

        try:
            client = HumeClient(api_key=self._api_key)

            voice_id = self._voice if self._voice else _DEFAULT_VOICE_ID

            start = time.monotonic()

            response = client.tts.synthesize_file_streaming(
                utterances=[
                    PostedUtterance(
                        text=text,
                        voice=PostedUtteranceVoiceWithId(
                            id=voice_id,
                            provider="HUME_AI",
                        ),
                    )
                ],
                format=FormatWav(),
                num_generations=1,
                instant_mode=True,
            )

            for chunk in response:
                if self._is_audio_chunk(chunk):
                    if ttfa_ms is None:
                        ttfa_ms = (time.monotonic() - start) * 1000
                        logger.debug(
                            "hume_ttfa",
                            model=self._model,
                            ttfa_ms=ttfa_ms,
                        )
                    audio_chunks.append(chunk)

        except Exception as exc:
            logger.debug("hume_error", exc_info=True)
            return TTSResult(
                provider="hume",
                model=self._model,
                voice=self._voice,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_path: Path | None = None
        if audio_chunks:
            audio_path = _write_raw(audio_chunks)

        return TTSResult(
            provider="hume",
            model=self._model,
            voice=self._voice,
            ttfa_ms=ttfa_ms,
            audio_path=audio_path,
            error=None,
        )


def _write_raw(chunks: list[bytes]) -> Path:
    """Write Hume audio chunks (already WAV-formatted) to a temp file."""
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(tmp_name, "wb") as fh:
        for chunk in chunks:
            fh.write(chunk)
    return Path(tmp_name)
