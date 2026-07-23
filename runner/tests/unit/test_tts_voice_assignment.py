# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the balanced TTS voice split (``_assign_tts_voices``)."""

from __future__ import annotations

from collections import Counter

from coval_bench.registries import MODEL_REGISTRY, Benchmark, ModelStatus, RegisteredModel
from coval_bench.runner.orchestrator import _assign_tts_voices


def _entry(voices: tuple[str, ...] = ()) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.TTS,
        provider="fake",
        model="fake-tts",
        voice="pinned",
        voices=voices,
        status=ModelStatus.ACTIVE,
    )


def test_even_split_exact() -> None:
    """10 items over a 2-voice pool always land exactly 5/5, whatever the run."""
    entry = _entry(("female", "male"))
    for run_id in range(1, 50):
        counts = Counter(_assign_tts_voices(entry, 10, run_id))
        assert counts == {"female": 5, "male": 5}


def test_uneven_pool_split() -> None:
    """When the pool doesn't divide the item count, voice counts differ by at most one."""
    entry = _entry(("f1", "f2", "m1", "m2"))
    counts = Counter(_assign_tts_voices(entry, 10, run_id=7))
    assert sum(counts.values()) == 10
    assert max(counts.values()) - min(counts.values()) <= 1

    entry = _entry(("female", "male"))
    counts = Counter(_assign_tts_voices(entry, 9, run_id=7))
    assert sorted(counts.values()) == [4, 5]


def test_same_run_reproducible() -> None:
    """The same run_id always produces the identical assignment."""
    entry = _entry(("female", "male"))
    assert _assign_tts_voices(entry, 10, run_id=42) == _assign_tts_voices(entry, 10, run_id=42)


def test_different_runs_rotate() -> None:
    """Different run_ids change which item gets which voice, not just the counts."""
    entry = _entry(("female", "male"))
    assignments = {tuple(_assign_tts_voices(entry, 10, run_id)) for run_id in range(1, 6)}
    assert len(assignments) > 1
    first_item_voices = {_assign_tts_voices(entry, 10, run_id)[0] for run_id in range(1, 20)}
    assert first_item_voices == {"female", "male"}


def test_no_pool_falls_back_to_pin() -> None:
    """Entries without a ``voices`` pool keep the single pinned ``voice`` for every item."""
    entry = _entry()
    assert _assign_tts_voices(entry, 10, run_id=1) == ["pinned"] * 10


def test_registry_pools_are_f_m_pairs() -> None:
    """Every ``voices`` pool is a distinct (female, male) pair on an active TTS entry."""
    pooled = [m for m in MODEL_REGISTRY if m.voices]
    assert pooled, "expected voice pools in the registry"
    for m in pooled:
        assert m.benchmark is Benchmark.TTS, f"{m.provider}/{m.model}: pool on non-TTS entry"
        assert m.status is ModelStatus.ACTIVE, f"{m.provider}/{m.model}: pool on inactive entry"
        assert len(m.voices) == 2, f"{m.provider}/{m.model}: pool must be a (female, male) pair"
        assert len(set(m.voices)) == 2, f"{m.provider}/{m.model}: duplicate voice in pool"


def test_active_tts_entries_have_pools() -> None:
    """All ACTIVE TTS models split voices, except providers with no per-gender voices:
    deepgram (voice IS the model string) and palabra (quality tiers only)."""
    missing = [
        (m.provider, m.model)
        for m in MODEL_REGISTRY
        if m.benchmark is Benchmark.TTS
        and m.status is ModelStatus.ACTIVE
        and not m.voices
        and m.provider not in ("deepgram", "palabra")
    ]
    assert missing == []
