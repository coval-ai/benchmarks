# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared ABC stubs for STT and TTS providers.

Do not add provider-specific logic here.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import structlog
from pydantic import SecretStr

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
    finalization_latency_seconds: float | None = None
    finalization_trigger: str | None = None
    final_audio_window_end_seconds: float | None = None
    finalization_warning_code: str | None = None
    finalization_timed_out: bool = False

    # Content
    first_token_content: str | None = None
    complete_transcript: str | None = None
    partial_transcripts: list[str] = field(default_factory=list)

    # Transcript metrics
    transcript_length: int | None = None
    word_count: int | None = None

    # Error handling
    error: str | None = None

    # Internal timing
    audio_start_time: float | None = None
    finalization_start_time: float | None = None

    # VAD events, for the providers that emit them
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
    # Upstream HTTP status, when the provider reported one. Apart from ``error`` so
    # callers can tell "out of credits" from "hiccup" without parsing prose. None means
    # no status was surfaced (websocket frames, SDK exceptions), not success.
    status_code: int | None = None
    http_version: str | None = None
    submit_to_headers_ms: float | None = None
    connection_reused: bool | None = None
    # The leading-silence part of ttfa_ms; None when offset detection didn't
    # run or failed, in which case ttfa_ms is arrival-only and has no split.
    leading_silence_ms: float | None = None


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
    """Abstract base class for speech-to-text providers.

    Subclasses implement :meth:`_transcribe`; :meth:`measure_ttft` owns the
    result object, the input guard, the total-time clock and the failure log.
    Streaming WebSocket providers should extend
    :class:`coval_bench.providers.stt._streaming.WebSocketSTTProvider` instead.
    """

    # Set when the wire protocol accepts exactly one input rate.
    _SAMPLE_RATE: ClassVar[int | None] = None

    def __init__(self, settings: Settings, model: str) -> None:
        self._settings = settings
        self._model = model
        self._log = structlog.get_logger(type(self).__module__).bind(
            provider=self.key(), model=model
        )

    @classmethod
    def key(cls) -> str:
        """Registry key: the provider's module name (``stt/soniox.py`` -> ``soniox``)."""
        return cls.__module__.rsplit(".", 1)[-1]

    @property
    def model(self) -> str:
        return self._model

    def _require_secret(self, field_name: str) -> SecretStr:
        value: SecretStr | None = getattr(self._settings, field_name, None)
        if value is None or not value.get_secret_value().strip():
            raise ValueError(f"{field_name} is required for the {self.key()} STT provider")
        return value

    def _require_setting(self, field_name: str) -> str:
        value: str | None = getattr(self._settings, field_name, None)
        if not value or not value.strip():
            raise ValueError(f"{field_name} is required for the {self.key()} STT provider")
        return value

    def _new_result(self) -> TranscriptionResult:
        return TranscriptionResult(provider=self.name)

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        """Measure TTFT (Time to First Token) for this provider.

        Args:
            audio_data: Raw PCM bytes (no WAV header).
            channels: Number of audio channels (1 = mono).
            sample_width: Bytes per sample (2 for PCM_16).
            sample_rate: Samples per second (e.g. 16 000).
            realtime_resolution: Chunk duration in seconds for simulated real-time
                streaming.

        Returns:
            A :class:`TranscriptionResult` with timing and transcript fields
            populated.
        """
        result = self._new_result()
        if channels != 1 or sample_width != 2:
            result.error = (
                f"{self.key()} requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result
        if self._SAMPLE_RATE is not None and sample_rate != self._SAMPLE_RATE:
            result.error = (
                f"{self.key()} requires {self._SAMPLE_RATE / 1000:g} kHz PCM input; "
                f"got {sample_rate} Hz"
            )
            return result
        if realtime_resolution <= 0:
            result.error = "realtime_resolution must be > 0"
            return result

        total_start = time.monotonic()
        try:
            await self._transcribe(audio_data, sample_rate, realtime_resolution, result)
        except Exception as exc:
            self._log.warning("stt_measure_ttft_failed", exc_info=exc)
            if result.error is None:
                result.error = str(exc)
        result.total_time = time.monotonic() - total_start
        return result

    @abstractmethod
    async def _transcribe(
        self,
        audio_data: bytes,
        sample_rate: int,
        realtime_resolution: float,
        result: TranscriptionResult,
    ) -> None:
        """Stream mono PCM16 *audio_data* and fill *result*. Raise to fail the clip."""


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
