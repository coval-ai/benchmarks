# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TTFA (Time-To-First-Audio) metric."""

from __future__ import annotations

import math

import numpy as np
import pytest

from coval_bench.metrics.ttfa import compute_ttfa, first_audible_offset_ms


def _silence_pcm(duration_ms: float, sample_rate: int, amplitude: float = 0.0) -> bytes:
    """Return *duration_ms* of (optionally low-noise) silence as int16 PCM bytes."""
    n = round(duration_ms / 1000.0 * sample_rate)
    if amplitude == 0.0:
        return np.zeros(n, dtype="<i2").tobytes()
    noise = (np.full(n, amplitude) * 32767.0).astype("<i2")
    return noise.tobytes()


def _tone_pcm(
    duration_ms: float, sample_rate: int, amplitude: float = 0.3, freq: float = 220.0
) -> bytes:
    """Return a *duration_ms* sine tone as int16 PCM bytes (amplitude in [0, 1])."""
    n = round(duration_ms / 1000.0 * sample_rate)
    t = np.arange(n) / sample_rate
    wave = amplitude * np.sin(2.0 * math.pi * freq * t)
    return (wave * 32767.0).astype("<i2").tobytes()


def test_ttfa_basic() -> None:
    """Known timestamps produce the correct millisecond value."""
    result = compute_ttfa(1000.0, 1000.5)
    assert result == pytest.approx(500.0)


def test_ttfa_zero_elapsed() -> None:
    """Start and end at the same instant → 0 ms."""
    assert compute_ttfa(5.0, 5.0) == pytest.approx(0.0)


def test_ttfa_large_value() -> None:
    """Larger elapsed times are scaled correctly."""
    result = compute_ttfa(0.0, 2.5)
    assert result == pytest.approx(2500.0)


def test_ttfa_fractional_milliseconds() -> None:
    """Sub-millisecond precision is preserved."""
    result = compute_ttfa(100.0, 100.0005)
    assert result == pytest.approx(0.5)


def test_ttfa_raises_on_reversed_timestamps() -> None:
    """ValueError when first_audio_chunk_at < audio_synthesis_start."""
    with pytest.raises(ValueError, match="first_audio_chunk_at must be >= audio_synthesis_start"):
        compute_ttfa(100.0, 99.9)


def test_ttfa_raises_strictly_negative_delta() -> None:
    """Even a tiny reversal raises."""
    with pytest.raises(ValueError):
        compute_ttfa(1.0, 0.9999999)


def test_ttfa_returns_float() -> None:
    result = compute_ttfa(10.0, 11.0)
    assert isinstance(result, float)


def test_ttfa_adds_leading_silence_offset() -> None:
    """Perceived TTFA = chunk-arrival latency + leading-silence offset."""
    assert compute_ttfa(1000.0, 1000.5, 50.0) == pytest.approx(550.0)


def test_ttfa_raises_on_negative_offset() -> None:
    with pytest.raises(ValueError, match="leading_silence_offset_ms must be >= 0"):
        compute_ttfa(1.0, 2.0, -1.0)


# --- first_audible_offset_ms ------------------------------------------------


def test_offset_pure_silence_returns_none() -> None:
    """All-zero PCM has no audible sample."""
    assert first_audible_offset_ms(_silence_pcm(500, 24000), 24000) is None


def test_offset_sub_threshold_tone_returns_none() -> None:
    """A tone too quiet to be audible (below threshold) is not detected."""
    quiet = _tone_pcm(500, 24000, amplitude=0.005)
    assert first_audible_offset_ms(quiet, 24000) is None


@pytest.mark.parametrize("amplitude", [0.025, 0.05, 0.1, 0.3])
def test_offset_ignores_pure_dc(amplitude: float) -> None:
    """A constant DC bias well above the raw threshold is inaudible → None."""
    # Raw RMS equals the amplitude (> threshold) but the AC energy is zero, so
    # no audible sample should be reported regardless of how large the bias is.
    assert first_audible_offset_ms(_silence_pcm(300, 24000, amplitude=amplitude), 24000) is None


def test_offset_ignores_dc_before_tone() -> None:
    """Leading DC bias is skipped; the offset lands on the real tone onset."""
    sr = 24000
    pcm = _silence_pcm(100, sr, amplitude=0.05) + _tone_pcm(300, sr)
    offset = first_audible_offset_ms(pcm, sr)
    assert offset is not None
    assert offset == pytest.approx(100.0, abs=12.0)


def test_offset_empty_returns_none() -> None:
    assert first_audible_offset_ms(b"", 24000) is None


def test_offset_immediate_tone_near_zero() -> None:
    """A tone from the first sample is detected at ~0 ms."""
    offset = first_audible_offset_ms(_tone_pcm(300, 24000), 24000)
    assert offset is not None
    assert offset == pytest.approx(0.0, abs=10.0)


@pytest.mark.parametrize("sample_rate", [22050, 24000, 48000])
def test_offset_silence_then_tone_at_known_offset(sample_rate: int) -> None:
    """Detected offset matches the leading-silence duration within a frame."""
    lead_ms = 200.0
    pcm = _silence_pcm(lead_ms, sample_rate) + _tone_pcm(300, sample_rate)
    offset = first_audible_offset_ms(pcm, sample_rate)
    assert offset is not None
    assert offset == pytest.approx(lead_ms, abs=12.0)


def test_offset_rejects_bad_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate must be > 0"):
        first_audible_offset_ms(_tone_pcm(100, 24000), 0)


def test_offset_rejects_misaligned_pcm() -> None:
    """An odd-length (non-frame-aligned) buffer gets a clear error, not a numpy one."""
    with pytest.raises(ValueError, match="whole number of frames"):
        first_audible_offset_ms(b"\x00\x01\x02", 24000)
