# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared finalize step for TTS providers: perceived TTFA + WAV output."""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path

import structlog

from coval_bench.metrics import compute_ttfa, first_audible_offset_ms
from coval_bench.providers.base import TTSResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Mono signed 16-bit PCM: 2 bytes per sample.
_PCM16_FRAME_BYTES = 2


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
    """
    # Drop a dangling partial sample so neither offset detection (which rejects
    # frame-misaligned PCM) nor the WAV header chokes on a split final frame.
    remainder = len(pcm) % _PCM16_FRAME_BYTES
    if remainder:
        pcm = pcm[: len(pcm) - remainder]

    ttfa_ms: float | None = None
    if audio_synthesis_start is not None and first_audio_chunk_at is not None:
        offset_ms = _safe_offset_ms(pcm, sample_rate, provider, model) if pcm else None
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
    return TTSResult(
        provider=provider,
        model=model,
        voice=voice,
        ttfa_ms=ttfa_ms,
        audio_path=audio_path,
        error=error,
    )


def _safe_offset_ms(pcm: bytes, sample_rate: int, provider: str, model: str) -> float | None:
    """Leading-silence offset, swallowing any error to a ``None`` (arrival-only) result.

    The offset is a latency metric, not load-bearing for the audio; a failure here
    must never propagate out of ``synthesize`` and discard the synthesized WAV.
    """
    try:
        return first_audible_offset_ms(pcm, sample_rate)
    except Exception:
        logger.warning("tts_offset_failed", provider=provider, model=model, exc_info=True)
        return None


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
