# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared finalize step for TTS providers: perceived TTFA + WAV output."""

from __future__ import annotations

import math
import os
import tempfile
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import structlog

from coval_bench.metrics import compute_ttfa, first_audible_offset_ms
from coval_bench.providers.base import TTSResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Mono signed 16-bit PCM: 2 bytes per sample.
_PCM16_FRAME_BYTES = 2

# Stable contract — matched by the reason classifier and the alerting log metric.
_SILENT_FAILURE_PREFIX = "provider closed the stream without sending audio or an error"
_INAUDIBLE_AUDIO_ERROR = "provider audio remained below the audibility threshold"
_LAST_FRAMES_MAX_CHARS = 400


def finalize_tts_result(
    *,
    provider: str,
    model: str,
    voice: str,
    pcm: bytes,
    sample_rate: int,
    audio_synthesis_start: float | None,
    first_audio_chunk_at: float | None,
    error: str | None = None,
    http_version: str | None = None,
    submit_to_headers_ms: float | None = None,
    connection_reused: bool | None = None,
    last_frames: Sequence[str] | None = None,
) -> TTSResult:
    """Build a TTSResult with perceived TTFA from assembled PCM and timing.

    Perceived TTFA = (first chunk arrival - synthesis start) + leading silence.
    ``ttfa_ms`` is ``None`` when no audio arrived (timestamps unset). On the error
    path, callers pass ``pcm=b""`` so no WAV is written and TTFA falls back to
    arrival-only (no offset), matching today's behaviour.

    The offset is best-effort: computing it is a latency *metric*, never a reason
    to discard good audio. A trailing partial sample (a 16-bit frame split across
    the final stream frame) is dropped, and any failure in offset detection is
    swallowed and degrades TTFA to arrival-only — the WAV is still written either
    way.

    Empty ``pcm`` with no ``error`` is never a success: it means the receive loop
    ended without synthesising anything and without recognising why. Every provider
    reaches its final ``finalize_tts_result`` by falling out of that loop, so
    without this guard an error frame the provider *did* send but the integration
    failed to match is discarded, and the row lands on the orchestrator's generic
    "no TTFA produced". Naming the state here is what keeps it diagnosable, for
    every provider at once. ``last_frames`` carries the last non-audio frames seen,
    when the caller retains them, so the provider's own wording survives even
    though its schema was never matched.
    """
    # Drop a dangling partial sample so neither offset detection (which rejects
    # frame-misaligned PCM) nor the WAV header chokes on a split final frame.
    remainder = len(pcm) % _PCM16_FRAME_BYTES
    if remainder:
        pcm = pcm[: len(pcm) - remainder]

    # Audibility is a property of the audio alone, so it is judged whenever there is
    # audio — not only when both timestamps happen to be set. A provider that returned
    # PCM but left a timestamp unset would otherwise skip the check entirely and fall
    # through to the orchestrator's generic reason.
    offset_ms: float | None = None
    inaudible_audio = False
    if pcm:
        offset_ms, detection_ok = _safe_offset_ms(pcm, sample_rate, provider, model)
        inaudible_audio = offset_ms is None and detection_ok

    ttfa_ms: float | None = None
    if audio_synthesis_start is not None and first_audio_chunk_at is not None:
        ttfa_ms = compute_ttfa(
            audio_synthesis_start, first_audio_chunk_at, offset_ms if offset_ms is not None else 0.0
        )
        logger.debug(
            "tts_ttfa",
            provider=provider,
            model=model,
            arrival_ms=(first_audio_chunk_at - audio_synthesis_start) * 1000.0,
            offset_ms=offset_ms,
            ttfa_ms=ttfa_ms,
        )

    audio_path = _write_wav(pcm, sample_rate) if pcm else None
    if not pcm and error is None:
        error = _silent_failure_error(last_frames)
        logger.warning("tts_silent_failure", provider=provider, model=model, error=error)
    elif inaudible_audio and error is None:
        # Audio arrived but no frame cleared the audibility threshold. Without this the
        # null offset collapses into 0.0 and TTFA becomes pure arrival time — a
        # plausible number measured off silence, which then enters the aggregates.
        #
        # The wording claims only what a fixed RMS threshold can support: the audio was
        # below it, not definitively that speech is absent. Unusually quiet or whispered
        # output would land here too, so the peak and RMS are logged to make that case
        # recognisable rather than indistinguishable from true silence.
        error = _INAUDIBLE_AUDIO_ERROR
        ttfa_ms = None
        peak, rms = _level_diagnostics(pcm)
        logger.warning(
            "tts_inaudible_audio",
            provider=provider,
            model=model,
            bytes=len(pcm),
            peak_dbfs=peak,
            rms_dbfs=rms,
        )
    return TTSResult(
        provider=provider,
        model=model,
        voice=voice,
        ttfa_ms=ttfa_ms,
        audio_path=audio_path,
        error=error,
        http_version=http_version,
        submit_to_headers_ms=submit_to_headers_ms,
        connection_reused=connection_reused,
        leading_silence_ms=offset_ms if ttfa_ms is not None else None,
    )


def _silent_failure_error(last_frames: Sequence[str] | None) -> str:
    """Diagnostic for a stream that ended with no audio and no reported error.

    The prefix is a stable contract: the reason classifier and the Cloud Logging
    metric both match on it, so it must not be reworded casually. When the caller
    retained trailing frames they are appended verbatim — that text is the
    provider's own explanation, which is exactly what an unmatched schema loses.
    """
    if not last_frames:
        return _SILENT_FAILURE_PREFIX
    tail = " | ".join(frame.strip() for frame in last_frames if frame.strip())
    if not tail:
        return _SILENT_FAILURE_PREFIX
    return f"{_SILENT_FAILURE_PREFIX}; last frames: {tail[:_LAST_FRAMES_MAX_CHARS]}"


def _level_diagnostics(pcm: bytes) -> tuple[float | None, float | None]:
    """Peak and RMS of *pcm* in dBFS, or ``(None, None)`` when it can't be read.

    Recorded alongside an inaudible-audio failure so quiet-but-real speech can be told
    apart from true silence after the fact. Diagnostics only — never load-bearing, so
    any failure degrades to ``None`` rather than propagating.
    """
    try:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size == 0:
            return None, None
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples**2)))
        return _to_dbfs(peak), _to_dbfs(rms)
    except Exception:
        return None, None


def _to_dbfs(value: float) -> float | None:
    """Linear amplitude in [0, 1] as dBFS, or ``None`` for digital silence."""
    if value <= 0.0:
        return None
    return round(20.0 * math.log10(value), 1)


def _safe_offset_ms(
    pcm: bytes, sample_rate: int, provider: str, model: str
) -> tuple[float | None, bool]:
    """Leading-silence offset as ``(offset_ms, detection_ok)``.

    The offset is a latency metric, not load-bearing for the audio; a failure here
    must never propagate out of ``synthesize`` and discard the synthesized WAV.

    ``detection_ok`` separates two states that both yield a ``None`` offset and must
    not be conflated: detection ran and found no audible frame (the audio really is
    silent — a synthesis failure), versus detection itself crashed (our problem, and
    no reason to condemn the provider's audio).
    """
    try:
        return first_audible_offset_ms(pcm, sample_rate), True
    except Exception as exc:
        logger.warning("tts_offset_failed", provider=provider, model=model, exc_info=exc)
        return None, False


def _write_wav(pcm: bytes, sample_rate: int) -> Path:
    """Write assembled mono 16-bit PCM to a temp WAV file."""
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(tmp_name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return Path(tmp_name)
