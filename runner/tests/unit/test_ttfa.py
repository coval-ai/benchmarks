# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TTFA (Time-To-First-Audio) metric."""

from __future__ import annotations

import pytest

from coval_bench.metrics.ttfa import compute_ttfa


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
