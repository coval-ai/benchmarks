# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TTFT, audio-to-final, and RTF metrics."""

from __future__ import annotations

import pytest

from coval_bench.metrics.ttft import (
    compute_audio_to_final,
    compute_rtf,
    compute_ttft,
)

# ---------------------------------------------------------------------------
# compute_ttft
# ---------------------------------------------------------------------------


def test_ttft_basic() -> None:
    """Known timestamps produce the correct second value."""
    assert compute_ttft(100.0, 100.5) == pytest.approx(0.5)


def test_ttft_zero_elapsed() -> None:
    assert compute_ttft(5.0, 5.0) == pytest.approx(0.0)


def test_ttft_large_value() -> None:
    result = compute_ttft(0.0, 3.75)
    assert result == pytest.approx(3.75)


def test_ttft_raises_on_reversed_timestamps() -> None:
    """ValueError when first_token_at < audio_send_start."""
    with pytest.raises(ValueError, match="first_token_at must be >= audio_send_start"):
        compute_ttft(101.0, 100.0)


def test_ttft_raises_slightly_reversed() -> None:
    with pytest.raises(ValueError):
        compute_ttft(1.0, 0.9999)


def test_ttft_returns_float() -> None:
    assert isinstance(compute_ttft(0.0, 1.0), float)


# ---------------------------------------------------------------------------
# compute_audio_to_final
# ---------------------------------------------------------------------------


def test_audio_to_final_basic() -> None:
    assert compute_audio_to_final(0.0, 1.5) == pytest.approx(1.5)


def test_audio_to_final_zero_elapsed() -> None:
    assert compute_audio_to_final(10.0, 10.0) == pytest.approx(0.0)


def test_audio_to_final_raises_on_reversed() -> None:
    with pytest.raises(ValueError, match="final_transcript_at must be >= audio_send_start"):
        compute_audio_to_final(5.0, 4.9)


def test_audio_to_final_returns_float() -> None:
    assert isinstance(compute_audio_to_final(0.0, 2.0), float)


# ---------------------------------------------------------------------------
# compute_rtf
# ---------------------------------------------------------------------------


def test_rtf_faster_than_realtime() -> None:
    """1 s wall-clock for 2 s audio → RTF = 0.5."""
    assert compute_rtf(1.0, 2.0) == pytest.approx(0.5)


def test_rtf_spec_example() -> None:
    """Spec: compute_rtf(2.0, 4.0) == 0.5."""
    assert compute_rtf(2.0, 4.0) == pytest.approx(0.5)


def test_rtf_equal_to_realtime() -> None:
    assert compute_rtf(3.0, 3.0) == pytest.approx(1.0)


def test_rtf_slower_than_realtime() -> None:
    assert compute_rtf(6.0, 3.0) == pytest.approx(2.0)


def test_rtf_zero_duration_raises() -> None:
    """Spec: compute_rtf(2.0, 0.0) raises ValueError."""
    with pytest.raises(ValueError, match="audio_duration_seconds must be > 0"):
        compute_rtf(2.0, 0.0)


def test_rtf_negative_duration_raises() -> None:
    with pytest.raises(ValueError):
        compute_rtf(1.0, -1.0)


def test_rtf_returns_float() -> None:
    assert isinstance(compute_rtf(1.0, 2.0), float)
