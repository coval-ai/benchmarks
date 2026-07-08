# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RTF and TTFS metrics."""

from __future__ import annotations

import pytest

from coval_bench.metrics.ttft import compute_rtf, compute_ttfs

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


# ---------------------------------------------------------------------------
# compute_ttfs
# ---------------------------------------------------------------------------


def test_ttfs_basic() -> None:
    """audio_to_final 5.3 s, speech-end at 5.0 s → 0.3 s finalization."""
    assert compute_ttfs(5.3, 5.0) == pytest.approx(0.3)


def test_ttfs_zero() -> None:
    assert compute_ttfs(5.0, 5.0) == pytest.approx(0.0)


def test_ttfs_clamps_early_final_to_zero() -> None:
    """A final ahead of the end-of-speech anchor clamps to the 0.0 floor, not an error."""
    assert compute_ttfs(4.9, 5.0) == pytest.approx(0.0)
    assert compute_ttfs(3.0, 5.0) == pytest.approx(0.0)


def test_ttfs_returns_float() -> None:
    assert isinstance(compute_ttfs(2.0, 1.0), float)
