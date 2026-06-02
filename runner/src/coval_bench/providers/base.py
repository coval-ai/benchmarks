# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared ABC stubs for STT and TTS providers.

providers-stt and providers-tts agents will refine these contracts when they
land.  Do not add provider-specific logic here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from coval_bench.config import Settings

# ---------------------------------------------------------------------------
# Shared result types
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Result of a single STT transcription request."""

    provider: str

    # Core timing measurements
    ttft_seconds: float | None = None
    total_time: float | None = None
    audio_to_final_seconds: float | None = None
    rtf_value: float | None = None

    # Content
    first_token_content: str | None = None
    complete_transcript: str | None = None
    partial_transcripts: list[str] = field(default_factory=list)

    # Transcript metrics
    transcript_length: int | None = None
    word_count: int | None = None
    wer_percentage: float | None = None

    # Error handling
    error: str | None = None

    # Internal timing
    audio_start_time: float | None = None

    # Deepgram-specific VAD data
    vad_first_detected: float | None = None
    vad_events_count: int | None = None
    vad_first_event_content: str | None = None


@dataclass
class TTSResult:
    """Result of a single TTS synthesis request."""

    provider: str
    model: str
    voice: str
    ttfa_ms: float | None
    audio_path: Path | None
    error: str | None
    http_version: str | None = None
    submit_to_headers_ms: float | None = None
    connection_reused: bool | None = None


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class Provider(ABC):
    """Base class for all benchmark providers (STT and TTS)."""

    _VALID_MODELS: ClassVar[frozenset[str]] = frozenset()

    def _model_supported(self, model: str) -> bool:
        """Return True when *model* is allowed for this provider.

        An empty ``_VALID_MODELS`` accepts any model. Providers that validate by
        pattern rather than an exact set override this method.
        """
        return not self._VALID_MODELS or model in self._VALID_MODELS

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider identifier, e.g. ``'deepgram-nova-2'``."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier used for this provider instance."""

    @classmethod
    async def warmup(cls, settings: Settings) -> None:
        """Optional pre-t0 setup, invoked once per run before the dataset loop.

        Default: no-op. Caller handles exceptions
        (``return_exceptions=True``), so a failed warmup never aborts the
        run. Implementers document protocol-specific behaviour on the
        override.
        """
        return None


class STTProvider(Provider, ABC):
    """Abstract base class for speech-to-text providers."""

    @abstractmethod
    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
        audio_duration: float | None = None,
    ) -> TranscriptionResult:
        """Measure TTFT (Time to First Token) for this provider.

        Args:
            audio_data: Raw PCM bytes (no WAV header).
            channels: Number of audio channels (1 = mono).
            sample_width: Bytes per sample (2 for PCM_16).
            sample_rate: Samples per second (e.g. 16 000).
            realtime_resolution: Chunk duration in seconds for simulated real-time
                streaming.
            audio_duration: Total audio duration in seconds; derived from audio_data
                length when ``None``.

        Returns:
            A :class:`TranscriptionResult` with timing and transcript fields
            populated.
        """


class TTSProvider(Provider, ABC):
    """Abstract base class for text-to-speech providers."""

    @abstractmethod
    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize *text* to speech and measure TTFA.

        Args:
            text: The prompt to synthesize.

        Returns:
            A :class:`TTSResult` with timing and audio path fields populated.
        """
