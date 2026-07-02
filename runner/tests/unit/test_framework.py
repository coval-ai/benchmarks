# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the standardized dataset build framework (framework.py).

Pure: no network, no GCS, no real audio. Focus on the generic balanced_sample
selector — the shared "balancing" logic every dataset relies on.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import cast

import pytest

from coval_bench.datasets.scripts.framework import Clip, _clean, balanced_sample


def _clip(
    sid: int,
    gender: str,
    native: bool | None,
    *,
    dur: float = 3.0,
    transcript: str = "what time is it now",
) -> Clip:
    return Clip(
        audio_path=Path(f"/fake/{sid}.flac"),
        transcript=transcript,
        meta={"slurp_id": sid, "gender": gender, "native": native},
        duration_sec=dur,
    )


def _sid(clip: Clip) -> int:
    return cast(int, clip.meta["slurp_id"])


def _gender(clip: Clip) -> str | None:
    g = clip.meta["gender"]
    return g if isinstance(g, str) and g in ("F", "M") else None


def _native(clip: Clip) -> bool | None:
    return cast("bool | None", clip.meta["native"])


def test_balanced_sample_dedups_by_key() -> None:
    """Duplicate dedup keys collapse to a single clip."""
    clips = [_clip(1, "F", True), _clip(1, "M", False), _clip(2, "M", False)]
    out = balanced_sample(clips, num=2, dedup_key=_sid, balance_dims=[])
    assert sorted(_sid(c) for c in out) == [1, 2]


def test_balanced_sample_excludes_untagged() -> None:
    """Clips untagged on a balance dim never enter the result."""
    clips = [_clip(i, "F", True) for i in range(10)] + [_clip(100, "UNK", None)]
    out = balanced_sample(clips, num=5, dedup_key=_sid, balance_dims=[_gender, _native])
    assert all(_gender(c) is not None and _native(c) is not None for c in out)
    assert 100 not in [_sid(c) for c in out]


def test_balanced_sample_stratified_balance() -> None:
    """Even round-robin over the dim cross-product balances each dimension."""
    clips: list[Clip] = []
    sid = 0
    for gender in ("F", "M"):
        for native in (True, False):
            for _ in range(20):
                clips.append(_clip(sid, gender, native))
                sid += 1
    out = balanced_sample(clips, num=40, dedup_key=_sid, balance_dims=[_gender, _native])
    genders = collections.Counter(_gender(c) for c in out)
    natives = collections.Counter(_native(c) for c in out)
    assert genders["F"] == genders["M"] == 20
    assert natives[True] == natives[False] == 20


def test_balanced_sample_aborts_below_floor() -> None:
    """A balance dim below the coverage floor aborts the build."""
    clips = [_clip(i, "F", True) for i in range(4)] + [
        _clip(100 + i, "UNK", True) for i in range(6)
    ]
    with pytest.raises(ValueError, match="below floor"):
        balanced_sample(clips, num=2, dedup_key=_sid, balance_dims=[_gender])


def test_balanced_sample_raises_if_insufficient() -> None:
    """Too few tagged clips to reach num aborts rather than under-filling."""
    clips = [_clip(i, "F", True) for i in range(3)]
    with pytest.raises(ValueError, match="need"):
        balanced_sample(clips, num=10, dedup_key=_sid, balance_dims=[_gender])


def test_balanced_sample_deterministic() -> None:
    """Result is independent of input order."""
    clips = [_clip(i, "F" if i % 2 else "M", i % 3 == 0) for i in range(60)]
    forward = balanced_sample(clips, num=10, dedup_key=_sid, balance_dims=[_gender])
    reverse = balanced_sample(list(reversed(clips)), num=10, dedup_key=_sid, balance_dims=[_gender])
    assert [_sid(c) for c in forward] == [_sid(c) for c in reverse]


def test_clean_filters_duration_band_and_word_floor() -> None:
    """_clean drops clips outside the duration band or under the word floor."""
    clips = [
        _clip(1, "F", True, dur=1.0),  # too short
        _clip(2, "F", True, dur=3.0),  # kept
        _clip(3, "F", True, dur=3.0, transcript="hi"),  # too few words
    ]
    out = _clean(clips, dur_min=2.0, dur_max=10.0, min_words=3)
    assert [_sid(c) for c in out] == [2]
