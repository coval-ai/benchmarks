# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for arena model pairing (uniform strategy)."""

from __future__ import annotations

import random

import pytest

from coval_bench.arena.pairing import active_tts_models, select_pair
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import ModelStatus, RegisteredModel


def _model(name: str) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.TTS,
        provider=name,
        model=name,
        voice="v",
        status=ModelStatus.ACTIVE,
    )


_MODELS = [_model("a"), _model("b"), _model("c")]


def test_select_pair_returns_two_distinct() -> None:
    first, second = select_pair(_MODELS, rng=random.Random(1))
    assert first in _MODELS
    assert second in _MODELS
    assert first is not second


def test_select_pair_requires_two_models() -> None:
    with pytest.raises(ValueError, match="at least two"):
        select_pair(_MODELS[:1])


def test_select_pair_is_deterministic_with_seed() -> None:
    assert select_pair(_MODELS, rng=random.Random(7)) == select_pair(_MODELS, rng=random.Random(7))


def test_active_tts_models_are_tts_and_active() -> None:
    roster = active_tts_models()
    assert len(roster) >= 2
    assert all(m.benchmark is Benchmark.TTS and m.status is ModelStatus.ACTIVE for m in roster)
