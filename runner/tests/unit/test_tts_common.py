# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared TTS finalize helper (perceived TTFA + WAV output)."""

from __future__ import annotations

import math
import wave
from unittest.mock import patch

import numpy as np
import pytest

from coval_bench.providers.tts._common import finalize_tts_result


def _silence_pcm(duration_ms: float, sample_rate: int) -> bytes:
    """Return *duration_ms* of pure silence as int16 PCM bytes."""
    n = round(duration_ms / 1000.0 * sample_rate)
    return np.zeros(n, dtype="<i2").tobytes()


def _tone_pcm(
    duration_ms: float, sample_rate: int, amplitude: float = 0.3, freq: float = 220.0
) -> bytes:
    """Return a *duration_ms* sine tone as int16 PCM bytes (amplitude in [0, 1])."""
    n = round(duration_ms / 1000.0 * sample_rate)
    t = np.arange(n) / sample_rate
    wave_arr = amplitude * np.sin(2.0 * math.pi * freq * t)
    return (wave_arr * 32767.0).astype("<i2").tobytes()


def test_finalize_adds_leading_silence_offset() -> None:
    """Silence-then-tone PCM → ttfa = arrival + offset, strictly greater than arrival."""
    sr = 24000
    lead_ms = 200.0
    pcm = _silence_pcm(lead_ms, sr) + _tone_pcm(300, sr)
    arrival_ms = 500.0

    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=pcm,
        sample_rate=sr,
        audio_synthesis_start=1000.0,
        first_audio_chunk_at=1000.0 + arrival_ms / 1000.0,
    )

    assert result.ttfa_ms is not None
    assert result.ttfa_ms > arrival_ms
    assert result.ttfa_ms == pytest.approx(arrival_ms + lead_ms, abs=12.0)

    assert result.audio_path is not None
    assert result.audio_path.exists()
    with wave.open(str(result.audio_path), "rb") as wav_file:
        assert wav_file.getframerate() == sr
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
    result.audio_path.unlink()


def test_finalize_immediate_tone_offset_near_zero() -> None:
    """Immediate-tone PCM → ttfa ≈ arrival (offset ≈ 0)."""
    sr = 24000
    arrival_ms = 300.0
    pcm = _tone_pcm(300, sr)

    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=pcm,
        sample_rate=sr,
        audio_synthesis_start=2.0,
        first_audio_chunk_at=2.0 + arrival_ms / 1000.0,
    )

    assert result.ttfa_ms is not None
    assert result.ttfa_ms == pytest.approx(arrival_ms, abs=10.0)
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_finalize_error_path_arrival_only() -> None:
    """Error path (empty pcm, timestamps set) → ttfa = arrival only, no WAV."""
    arrival_ms = 420.0
    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=b"",
        sample_rate=24000,
        audio_synthesis_start=10.0,
        first_audio_chunk_at=10.0 + arrival_ms / 1000.0,
        error="boom",
    )

    assert result.ttfa_ms == pytest.approx(arrival_ms)
    assert result.audio_path is None
    assert result.error == "boom"


def test_finalize_odd_length_pcm_still_writes_wav() -> None:
    """A dangling partial sample (split final frame) is dropped, not fatal.

    Regression: offset detection rejects frame-misaligned PCM. The finalize step
    must align it, still detect the leading-silence offset, and write the WAV.
    """
    sr = 24000
    lead_ms = 200.0
    pcm = _silence_pcm(lead_ms, sr) + _tone_pcm(300, sr) + b"\x07"  # stray odd byte

    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=pcm,
        sample_rate=sr,
        audio_synthesis_start=1.0,
        first_audio_chunk_at=1.5,
    )

    assert result.ttfa_ms is not None
    assert result.ttfa_ms == pytest.approx(500.0 + lead_ms, abs=12.0)
    assert result.audio_path is not None
    assert result.audio_path.exists()
    with wave.open(str(result.audio_path), "rb") as wav_file:
        # The dangling byte was dropped → frame count is a whole number of samples.
        assert wav_file.getnframes() == (len(pcm) - 1) // 2
    result.audio_path.unlink()


def test_finalize_offset_failure_falls_back_and_writes_wav() -> None:
    """If offset detection raises, TTFA degrades to arrival-only and the WAV still writes.

    A latency-metric failure must never discard good synthesized audio.
    """
    sr = 24000
    pcm = _tone_pcm(300, sr)
    arrival_ms = 250.0

    with patch(
        "coval_bench.providers.tts._common.first_audible_offset_ms",
        side_effect=RuntimeError("librosa boom"),
    ):
        result = finalize_tts_result(
            provider="test",
            model="m",
            voice="v",
            pcm=pcm,
            sample_rate=sr,
            audio_synthesis_start=10.0,
            first_audio_chunk_at=10.0 + arrival_ms / 1000.0,
        )

    assert result.ttfa_ms == pytest.approx(arrival_ms)  # arrival only, offset dropped
    assert result.error is None
    assert result.audio_path is not None
    assert result.audio_path.exists()
    result.audio_path.unlink()


def test_finalize_no_audio_returns_none() -> None:
    """No audio (first_audio_chunk_at unset) → ttfa None, no WAV."""
    result = finalize_tts_result(
        provider="test",
        model="m",
        voice="v",
        pcm=b"",
        sample_rate=24000,
        audio_synthesis_start=10.0,
        first_audio_chunk_at=None,
    )

    assert result.ttfa_ms is None
    assert result.audio_path is None
    assert result.error is None
