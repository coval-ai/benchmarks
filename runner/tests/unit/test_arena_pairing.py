# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for arena model pairing (adaptive + cold-start strategies)."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from coval_bench.arena.pairing import active_tts_models, select_pair
from coval_bench.db.models import PairingRating
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


def _rating(elo: float, ci: float | None) -> PairingRating:
    return PairingRating(rating_elo=elo, ci_half_width=ci)


def _key(m: RegisteredModel) -> tuple[str, str]:
    return (m.provider, m.model)


_MODELS = [_model("a"), _model("b"), _model("c")]


def test_select_pair_returns_two_distinct() -> None:
    first, second = select_pair(_MODELS, {}, rng=random.Random(1))
    assert first in _MODELS
    assert second in _MODELS
    assert first is not second


def test_select_pair_requires_two_models() -> None:
    with pytest.raises(ValueError, match="at least two"):
        select_pair(_MODELS[:1], {})


def test_cold_start_is_deterministic_with_seed() -> None:
    assert select_pair(_MODELS, {}, rng=random.Random(7)) == select_pair(
        _MODELS, {}, rng=random.Random(7)
    )


def test_adaptive_is_deterministic_with_seed() -> None:
    ratings = {
        _key(m): _rating(e, 30.0) for m, e in zip(_MODELS, [1000.0, 1100.0, 1200.0], strict=True)
    }
    assert select_pair(_MODELS, ratings, rng=random.Random(7)) == select_pair(
        _MODELS, ratings, rng=random.Random(7)
    )


def test_close_pair_outscores_blowout() -> None:
    # a~b are near-ties, c is a far outlier; the near-tie should dominate draws.
    ratings = {
        _key(m): _rating(e, 50.0) for m, e in zip(_MODELS, [1000.0, 1005.0, 1400.0], strict=True)
    }
    counts: Counter[frozenset[str]] = Counter(
        frozenset((a.model, b.model))
        for a, b in (select_pair(_MODELS, ratings, rng=random.Random(i)) for i in range(500))
    )
    assert counts[frozenset(("a", "b"))] > counts[frozenset(("a", "c"))]


def test_high_ci_model_oversampled_at_equal_gap() -> None:
    # equal ratings, but c is far more uncertain → it should appear more often.
    ratings = {
        _key(m): _rating(1000.0, ci) for m, ci in zip(_MODELS, [10.0, 10.0, 200.0], strict=True)
    }
    counts: Counter[str] = Counter(
        m.model
        for pair in (select_pair(_MODELS, ratings, rng=random.Random(i)) for i in range(500))
        for m in pair
    )
    assert counts["c"] > counts["a"]


def test_new_model_absent_from_ratings_is_reachable() -> None:
    # c has no rating → treated as max uncertainty, so it must still get sampled.
    ratings = {_key(_MODELS[0]): _rating(1000.0, 20.0), _key(_MODELS[1]): _rating(1010.0, 20.0)}
    seen = {
        m.model
        for pair in (select_pair(_MODELS, ratings, rng=random.Random(i)) for i in range(200))
        for m in pair
    }
    assert "c" in seen


def test_all_converged_falls_back_to_uniform() -> None:
    # every model fully converged (ci=0) → weights sum to 0 → uniform fallback.
    ratings = {
        _key(m): _rating(e, 0.0) for m, e in zip(_MODELS, [1000.0, 1100.0, 1200.0], strict=True)
    }
    first, second = select_pair(_MODELS, ratings, rng=random.Random(3))
    assert first is not second


def test_active_tts_models_are_tts_and_active() -> None:
    roster = active_tts_models()
    assert len(roster) >= 2
    assert all(m.benchmark is Benchmark.TTS and m.status is ModelStatus.ACTIVE for m in roster)
