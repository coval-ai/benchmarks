# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for hf_source + build helpers (pure: no network)."""

from __future__ import annotations

from pathlib import Path

from coval_bench.datasets.scripts.build import _meta_dim
from coval_bench.datasets.scripts.framework import Clip
from coval_bench.datasets.scripts.hf_source import _as_duration


def _clip(meta: dict[str, object]) -> Clip:
    return Clip(audio_path=Path("/x.wav"), transcript="hi there now", meta=meta)


def test_as_duration_coerces_null_and_junk() -> None:
    """Null / blank / non-numeric duration cells become 0.0 instead of crashing."""
    assert _as_duration(3.5) == 3.5
    assert _as_duration("2.0") == 2.0
    assert _as_duration(None) == 0.0
    assert _as_duration("") == 0.0
    assert _as_duration("n/a") == 0.0


def test_meta_dim_keeps_false_and_zero() -> None:
    """Balancing on a boolean/numeric column keeps False/0 (only missing → untagged)."""
    native = _meta_dim("native")
    assert native(_clip({"native": False})) is False
    assert native(_clip({"native": True})) is True
    assert _meta_dim("count")(_clip({"count": 0})) == 0
    assert native(_clip({})) is None
    assert native(_clip({"native": ""})) is None
