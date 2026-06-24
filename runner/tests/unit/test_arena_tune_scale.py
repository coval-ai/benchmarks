# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the offline tune-scale simulation."""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from coval_bench.arena.pairing import active_tts_models
from coval_bench.arena.rating import _ELO_SCALE
from coval_bench.arena.tune_scale import (
    ScaleResult,
    _davidson,
    _penalized_ce,
    _sample_outcome,
    _true_strengths,
    tune_scale,
)


def _fast(seed: int) -> list[ScaleResult]:
    """tune_scale with fast knobs: quick but still exercises refit + pairing."""
    return tune_scale(
        scales=(100.0, 200.0),
        n_battles=120,
        refit_every=40,
        replications=1,
        bootstrap_rounds=10,
        seed=seed,
    )


def test_penalized_ce_at_coin_flip_is_ln2_with_no_penalty() -> None:
    # p=0.5 -> CE = ln2 exactly, and the K-term must not fire.
    assert math.isclose(_penalized_ce(0.5, k=1.0), math.log(2.0))


def test_penalized_ce_below_baseline_is_unpenalized() -> None:
    # Confident-and-right (p=0.9 > 0.5) -> CE < ln2 -> no penalty added.
    assert math.isclose(_penalized_ce(0.9, k=1.0), -math.log(0.9))


def test_penalized_ce_penalizes_confidently_wrong() -> None:
    # p=0.1 < 0.5 -> CE = -ln(0.1) > ln2 -> add k*(CE - ln2).
    ce = -math.log(0.1)
    assert math.isclose(_penalized_ce(0.1, k=1.0), ce + (ce - math.log(2.0)))
    # Larger K bites harder on the same wrong call.
    assert _penalized_ce(0.1, k=5.0) > _penalized_ce(0.1, k=1.0)


def test_penalized_ce_is_monotonic_in_confidence() -> None:
    assert _penalized_ce(0.9, k=1.0) < _penalized_ce(0.5, k=1.0) < _penalized_ce(0.1, k=1.0)


def test_davidson_probs_sum_to_one() -> None:
    pa, pb, ptie = _davidson(0.4, -0.2, 1.0)
    assert math.isclose(pa + pb + ptie, 1.0, rel_tol=1e-12)


def test_davidson_stronger_model_wins_more() -> None:
    pa, pb, _ = _davidson(1.0, -1.0, 1.0)
    assert pa > pb


def test_davidson_is_symmetric_under_swap() -> None:
    pa, pb, ptie = _davidson(0.7, 0.1, 1.5)
    pb2, pa2, ptie2 = _davidson(0.1, 0.7, 1.5)
    assert math.isclose(pa, pa2) and math.isclose(pb, pb2)
    assert math.isclose(ptie, ptie2)


def test_true_strengths_span_the_requested_elo_spread() -> None:
    roster = active_tts_models()
    theta = _true_strengths(roster, elo_spread=800.0, rng=random.Random(0))
    elo_range = (max(theta.values()) - min(theta.values())) * _ELO_SCALE
    assert math.isclose(elo_range, 800.0, rel_tol=1e-9)


def test_sample_outcome_follows_the_probabilities() -> None:
    rng = random.Random(0)
    counts = Counter(_sample_outcome(0.7, 0.2, rng) for _ in range(20_000))
    assert math.isclose(counts["A_WIN"] / 20_000, 0.7, abs_tol=0.02)
    assert math.isclose(counts["B_WIN"] / 20_000, 0.2, abs_tol=0.02)
    assert math.isclose(counts["TIE"] / 20_000, 0.1, abs_tol=0.02)


def test_tune_scale_is_deterministic_for_a_seed() -> None:
    a = _fast(seed=0)
    b = _fast(seed=0)
    assert [(r.scale, r.loss) for r in a] == [(r.scale, r.loss) for r in b]


def test_tune_scale_returns_one_finite_result_per_scale() -> None:
    results = _fast(seed=0)
    assert [r.scale for r in results] == [100.0, 200.0]
    assert all(math.isfinite(r.loss) for r in results)
    assert all(r.decisive > 0 for r in results)


def test_short_run_with_no_refit_is_still_finite() -> None:
    # n_battles < refit_every -> the fit never runs, so every battle is scored
    # against the neutral prior. Loss must stay finite (not collapse to inf).
    results = tune_scale(
        scales=(150.0,),
        n_battles=20,
        refit_every=100,
        replications=1,
        bootstrap_rounds=0,
        seed=0,
    )
    assert math.isfinite(results[0].loss)
    assert results[0].decisive > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scales": ()},
        {"n_battles": 0},
        {"refit_every": 0},
        {"replications": 0},
        {"bootstrap_rounds": -1},
    ],
)
def test_tune_scale_rejects_invalid_params(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        tune_scale(**kwargs)  # type: ignore[arg-type]
