# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud Text-to-Speech provider (gRPC StreamingSynthesize).

This provider is gated on the optional extra ``google-tts``:

    uv sync --extra google-tts

Auth: Application Default Credentials (the runner service account in Cloud Run).
Models: chirp-3-hd (bidirectional — audio streams while text streams in),
gemini-2.5-flash-tts (output streaming only — synthesis starts at input half-close).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Generator

import structlog

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

try:
    from google.cloud import texttospeech

    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False

SAMPLE_RATE = 24000

# One warm gRPC channel per process so TTFA never pays connection/auth cold start.
_client_lock = threading.Lock()
_shared_client: texttospeech.TextToSpeechClient | None = None


def _get_shared_client() -> texttospeech.TextToSpeechClient:
    global _shared_client
    with _client_lock:
        if _shared_client is None:
            _shared_client = texttospeech.TextToSpeechClient()
        return _shared_client


class GoogleTTSProvider(TTSProvider):
    """Google Cloud TTS over the bidirectional StreamingSynthesize RPC.

    Install with:  uv sync --extra google-tts
    Requires:      Application Default Credentials (the runner service account in Cloud Run).
    """

    _VALID_MODELS = frozenset({"chirp-3-hd", "gemini-2.5-flash-tts"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if not GOOGLE_TTS_AVAILABLE:
            raise ImportError(
                "google-cloud-texttospeech is not installed. "
                "Install it with: uv sync --extra google-tts"
            )
        self._model = model
        self._voice = voice

    @property
    def name(self) -> str:
        return f"google-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Open the shared gRPC channel and cache an auth token before t0."""
        if not GOOGLE_TTS_AVAILABLE:
            return
        t0 = time.monotonic()
        await asyncio.to_thread(lambda: _get_shared_client().list_voices(language_code="en-US"))
        logger.info("google_tts_prewarm", warmup_ms=round((time.monotonic() - t0) * 1000, 1))

    def _voice_params(self) -> texttospeech.VoiceSelectionParams:
        if self._model.startswith("gemini-"):
            return texttospeech.VoiceSelectionParams(
                name=self._voice,
                language_code="en-US",
                model_name=self._model,
            )
        # Chirp voice names embed the locale: en-US-Chirp3-HD-Kore.
        return texttospeech.VoiceSelectionParams(
            name=self._voice,
            language_code="-".join(self._voice.split("-")[:2]),
        )

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via StreamingSynthesize and return a TTSResult."""
        if not self._model_supported(self._model):
            return TTSResult(
                provider="google",
                model=self._model,
                voice=self._voice,
                ttfa_ms=None,
                audio_path=None,
                error=(
                    f"Unsupported Google TTS model: {self._model}. "
                    f"Valid: {sorted(self._VALID_MODELS)}"
                ),
            )

        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        streaming_config = texttospeech.StreamingSynthesizeConfig(
            voice=self._voice_params(),
            streaming_audio_config=texttospeech.StreamingAudioConfig(
                audio_encoding=texttospeech.AudioEncoding.PCM,
                sample_rate_hertz=SAMPLE_RATE,
            ),
        )

        def _requests() -> Generator[texttospeech.StreamingSynthesizeRequest, None, None]:
            yield texttospeech.StreamingSynthesizeRequest(streaming_config=streaming_config)
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=text)
            )
            # Generator exhaustion half-closes the stream (Gemini's synthesis trigger).

        def _run_sync() -> None:
            nonlocal start, first_chunk_at
            client = _get_shared_client()
            # t0 — synthesis trigger; the channel is warm, so no connect cost leaks in.
            start = time.monotonic()
            for response in client.streaming_synthesize(requests=_requests()):
                if response.audio_content:
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                    audio_chunks.append(bytes(response.audio_content))

        try:
            await asyncio.to_thread(_run_sync)
        except Exception as exc:
            logger.warning("google_tts_error", provider="google", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="google",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="google",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )
