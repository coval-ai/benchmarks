# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Davidson-BT arena rating engine.

Done-criteria from the implementation plan (layer #3):
  * synthetic data recovers a known ranking,
  * confidence intervals shrink as vote count grows,
  * adding ties moves the tie parameter.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from coval_bench.arena import rating
from coval_bench.arena.rating import (
    BattleOutcome,
    classify_status,
    compute_ratings,
)


def _simulate(
    true_theta: dict[str, float],
    *,
    nu: float,
    battles_per_pair: int,
    seed: int,
) -> list[BattleOutcome]:
    """Round-robin battles drawn from the Davidson model at known theta/nu."""
    rng = np.random.default_rng(seed)
    models = list(true_theta)
    outcomes: list[BattleOutcome] = []
    for a_idx in range(len(models)):
        for b_idx in range(a_idx + 1, len(models)):
            a, b = models[a_idx], models[b_idx]
            ti, tj = true_theta[a], true_theta[b]
            ei, ej = math.exp(ti), math.exp(tj)
            eij = math.exp((ti + tj) / 2.0)
            denom = ei + ej + nu * eij
            probs = [ei / denom, ej / denom, nu * eij / denom]
            draws = rng.choice(3, size=battles_per_pair, p=probs)
            for d in draws:
                outcome = ("A_WIN", "B_WIN", "TIE")[int(d)]
                outcomes.append(BattleOutcome(model_a=a, model_b=b, outcome=outcome))
    return outcomes


# ---------------------------------------------------------------------------
# 1. Recover a known ranking
# ---------------------------------------------------------------------------


def test_recovers_known_ranking() -> None:
    true_theta = {"m0": 1.2, "m1": 0.6, "m2": 0.0, "m3": -0.6, "m4": -1.2}
    outcomes = _simulate(true_theta, nu=0.3, battles_per_pair=400, seed=7)

    result = compute_ratings(outcomes, bootstrap_rounds=100, seed=1)

    recovered_order = [m.model_id for m in result.models]  # sorted by Elo desc
    expected_order = sorted(true_theta, key=lambda m: true_theta[m], reverse=True)
    assert recovered_order == expected_order

    # Elo should be monotonic in true strength and centered on the 1500 base.
    elo = {m.model_id: m.rating_elo for m in result.models}
    assert elo["m0"] > elo["m2"] > elo["m4"]
    mean_elo = sum(elo.values()) / len(elo)
    assert abs(mean_elo - 1500.0) < 1.0


# ---------------------------------------------------------------------------
# 2. CIs shrink with N
# ---------------------------------------------------------------------------


def test_ci_shrinks_with_more_votes() -> None:
    true_theta = {"m0": 0.8, "m1": 0.0, "m2": -0.8}

    small = compute_ratings(
        _simulate(true_theta, nu=0.3, battles_per_pair=30, seed=3),
        bootstrap_rounds=200,
        seed=1,
    )
    large = compute_ratings(
        _simulate(true_theta, nu=0.3, battles_per_pair=600, seed=3),
        bootstrap_rounds=200,
        seed=1,
    )

    small_hw = {m.model_id: m.ci_half_width for m in small.models}
    large_hw = {m.model_id: m.ci_half_width for m in large.models}
    for model_id in true_theta:
        assert large_hw[model_id] < small_hw[model_id]


# ---------------------------------------------------------------------------
# 3. Ties move the tie parameter
# ---------------------------------------------------------------------------


def test_ties_move_tie_parameter() -> None:
    true_theta = {"m0": 0.5, "m1": 0.0, "m2": -0.5}

    few_ties = compute_ratings(
        _simulate(true_theta, nu=0.05, battles_per_pair=300, seed=5),
        bootstrap_rounds=1,
        seed=1,
    )
    many_ties = compute_ratings(
        _simulate(true_theta, nu=3.0, battles_per_pair=300, seed=5),
        bootstrap_rounds=1,
        seed=1,
    )

    assert many_ties.tie_param > few_ties.tie_param
    # The tie-heavy fit should also report many more ties than the clean one.
    assert sum(m.ties for m in many_ties.models) > sum(m.ties for m in few_ties.models)


# ---------------------------------------------------------------------------
# Edge cases and unit-level checks
# ---------------------------------------------------------------------------


def test_empty_input() -> None:
    result = compute_ratings([], bootstrap_rounds=10, seed=1)
    assert result.models == []
    assert result.tie_param == 0.0


def test_separation_stays_finite() -> None:
    # A beats B every time -> MLE wants infinite theta; ridge must keep it finite.
    outcomes = [BattleOutcome(model_a="A", model_b="B", outcome="A_WIN") for _ in range(50)]
    result = compute_ratings(outcomes, bootstrap_rounds=20, seed=1)
    by_id = {m.model_id: m for m in result.models}
    assert by_id["A"].rating_elo > by_id["B"].rating_elo
    assert all(math.isfinite(m.rating_elo) for m in result.models)
    assert by_id["A"].wins == 50.0
    assert by_id["B"].losses == 50.0


def test_win_loss_tie_tallies() -> None:
    outcomes = [
        BattleOutcome(model_a="A", model_b="B", outcome="A_WIN"),
        BattleOutcome(model_a="A", model_b="B", outcome="B_WIN"),
        BattleOutcome(model_a="B", model_b="A", outcome="B_WIN"),  # model_b (A) wins again
        BattleOutcome(model_a="A", model_b="B", outcome="TIE"),
    ]
    result = compute_ratings(outcomes, bootstrap_rounds=10, seed=1)
    by_id = {m.model_id: m for m in result.models}
    assert by_id["A"].wins == 2.0
    assert by_id["A"].losses == 1.0
    assert by_id["A"].ties == 1
    assert by_id["A"].votes_total == 4
    assert by_id["B"].wins == 1.0
    assert by_id["B"].losses == 2.0


def test_classify_status() -> None:
    assert classify_status(None) == "preliminary"  # no CI -> floor
    assert classify_status(float("nan")) == "preliminary"
    assert classify_status(10.0) == "established"
    assert classify_status(30.0) == "established"  # boundary inclusive
    assert classify_status(45.0) == "usable"
    assert classify_status(60.0) == "usable"  # boundary inclusive
    assert classify_status(200.0) == "preliminary"  # CI too wide -> floor


def test_ci_is_well_formed() -> None:
    # Only the geometrically guaranteed properties are asserted. The point
    # estimate is NOT guaranteed to sit inside the bootstrap percentile band
    # (the MLE and the bootstrap distribution are distinct objects), so we do
    # not assert ci_low <= rating_elo <= ci_high.
    true_theta = {"m0": 0.7, "m1": 0.0, "m2": -0.7}
    result = compute_ratings(
        _simulate(true_theta, nu=0.3, battles_per_pair=200, seed=9),
        bootstrap_rounds=200,
        seed=1,
    )
    for m in result.models:
        assert m.ci_low is not None and m.ci_high is not None
        assert m.ci_low <= m.ci_high
        assert m.ci_half_width is not None and m.ci_half_width >= 0.0


def test_no_ci_when_bootstrap_disabled() -> None:
    # bootstrap_rounds=0 -> no resamples -> CI fields are None, never NaN.
    outcomes = [BattleOutcome(model_a="A", model_b="B", outcome="A_WIN") for _ in range(10)]
    result = compute_ratings(outcomes, bootstrap_rounds=0, seed=1)
    for m in result.models:
        assert m.ci_low is None
        assert m.ci_high is None
        assert m.ci_half_width is None
        assert m.status == "preliminary"


def test_main_fit_nonconvergence_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # If the optimizer reports failure, the main fit must refuse, not return junk.
    class _FailRes:
        success = False
        message = "iteration limit"
        x = np.zeros(3, dtype=np.float64)  # 2 models + eta

    monkeypatch.setattr(rating, "minimize", lambda *a, **k: _FailRes())
    outcomes = [BattleOutcome(model_a="A", model_b="B", outcome="A_WIN")]
    with pytest.raises(rating.ConvergenceError):
        compute_ratings(outcomes, bootstrap_rounds=0, seed=1)


def test_bootstrap_skips_nonconverged(monkeypatch: pytest.MonkeyPatch) -> None:
    # A resample that fails to converge is skipped, not fatal: _bootstrap returns
    # NaN (which compute_ratings later renders as None), without raising.
    def _always_fail(*_a: object, **_k: object) -> object:
        raise rating.ConvergenceError("forced")

    monkeypatch.setattr(rating, "_fit", _always_fail)
    rng = np.random.default_rng(0)
    outcomes = [BattleOutcome(model_a="A", model_b="B", outcome="A_WIN")]
    low, high = rating._bootstrap(outcomes, ["A", "B"], rounds=5, reg=0.1, rng=rng)
    assert np.all(np.isnan(low))
    assert np.all(np.isnan(high))


def test_self_battle_is_ignored() -> None:
    # A self-battle (model_a == model_b) carries no comparison signal and must
    # not corrupt the per-pair tallies or shift the ratings.
    real = [BattleOutcome(model_a="A", model_b="B", outcome="A_WIN") for _ in range(5)]
    polluted = real + [BattleOutcome(model_a="A", model_b="A", outcome="A_WIN") for _ in range(3)]

    clean = {m.model_id: m for m in compute_ratings(real, bootstrap_rounds=0, seed=1).models}
    mixed = {m.model_id: m for m in compute_ratings(polluted, bootstrap_rounds=0, seed=1).models}

    assert mixed["A"].wins == clean["A"].wins == 5.0
    assert mixed["A"].votes_total == clean["A"].votes_total == 5
    assert mixed["A"].rating_elo == pytest.approx(clean["A"].rating_elo)


def test_single_bootstrap_sample_gives_no_ci() -> None:
    # With one bootstrap round each model has at most one resample Elo. A lone
    # sample is not a usable interval, so CI must be None and the status the
    # preliminary floor -- not a false zero-width "established".
    outcomes = [BattleOutcome(model_a="A", model_b="B", outcome="A_WIN") for _ in range(10)]
    result = compute_ratings(outcomes, bootstrap_rounds=1, seed=1)
    for m in result.models:
        assert m.ci_low is None
        assert m.ci_high is None
        assert m.ci_half_width is None
        assert m.status == "preliminary"


def test_invalid_inputs_raise() -> None:
    outcomes = [BattleOutcome(model_a="A", model_b="B", outcome="A_WIN")]
    with pytest.raises(ValueError):
        compute_ratings(outcomes, bootstrap_rounds=-1, seed=1)
    with pytest.raises(ValueError):
        compute_ratings(outcomes, reg=-0.1, seed=1)
    with pytest.raises(ValueError):
        compute_ratings(outcomes, reg=float("inf"), seed=1)
