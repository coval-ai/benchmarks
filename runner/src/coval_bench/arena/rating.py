# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Voice Arena rating engine.

Pure functions: vote outcomes in, ratings out. No DB, no I/O. The only input is
a sequence of head-to-head battle outcomes (the shape of ``arena.votes`` joined
to ``arena.battles``); the output maps one-to-one onto ``arena.leaderboard_snapshots``.

Model
-----
Davidson (1970) extension of Bradley-Terry to allow ties. Each model has a
log-strength ``theta``; one shared tie parameter ``nu >= 0`` controls how often a
comparison ends in a draw. For a battle between ``i`` and ``j``::

    D            = exp(theta_i) + exp(theta_j) + nu * exp((theta_i + theta_j) / 2)
    P(i wins)    = exp(theta_i) / D
    P(j wins)    = exp(theta_j) / D
    P(tie)       = nu * exp((theta_i + theta_j) / 2) / D

``nu = 0`` collapses exactly to plain Bradley-Terry (ties impossible).

Fitting
-------
Maximum likelihood via L-BFGS-B with an analytic gradient. The score statistic
``wins + 0.5 * ties`` is sufficient for ``theta``; the *log-likelihood* gradient
is ``observed - expected`` of that score. The objective adds a small L2 ridge
(`reg`), so the fitted optimum satisfies ``observed - expected = 2 * reg * theta``
(a slight shrink toward 0), not exactly zero. The ridge also pins the otherwise
shift-invariant scale, tames separation (a model that wins or loses everything),
keeps bootstrap refits finite, and makes the objective strictly convex (unique
optimum). A fit that still fails to converge raises ``ConvergenceError`` rather
than return a non-MLE point.

Elo
---
``elo = 1500 + (400 / ln 10) * theta`` — the standard Bradley-Terry-to-Elo map, so
a 400-point gap means a 10:1 odds favourite (ties aside). Elo is a display rescale
of ``theta`` (``rating_bt``) and carries no information ``theta`` does not.

Confidence
----------
Percentile bootstrap over battles: resample battles with replacement, refit, and
take the 2.5/97.5 Elo percentiles. The CI half-width comes from the same draws.
More votes -> tighter interval. A resample whose fit fails to converge is skipped.

Assumptions
-----------
Ratings assume a *connected* comparison graph: each model must be linked to the
rest through a chain of battles, or cross-cluster strengths are undefined and the
ridge silently pins them. Connectivity is maintained upstream by adaptive pairing
(plan §10); a defensive connected-components guard in this engine is a follow-up.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class ConvergenceError(RuntimeError):
    """Raised when the Davidson MLE optimizer fails to reach the optimum.

    The fitted theta/nu are only the maximum-likelihood estimate when the solver
    converges (gradient ~ 0). On failure the returned point is an arbitrary spot
    on the likelihood surface, so we refuse it rather than publish wrong ratings.
    """


# Bump on any behavioural change to the rating math (model, Elo map, bootstrap,
# status thresholds). Persisted on leaderboard snapshots so a stored rating can
# be tied back to the method that produced it. The numeric suffix is zero-padded
# so its lexicographic order (used as the leaderboard tiebreaker on equal
# timestamps) still matches numeric order past version 9.
METHODOLOGY_VERSION: Literal["davidson-bt-001"] = "davidson-bt-001"

# Standard Bradley-Terry -> Elo scale: a difference of 400 Elo == 10:1 win odds.
_ELO_BASE = 1500.0
_ELO_SCALE = 400.0 / math.log(10.0)

# nu is optimized on the log scale (eta = log nu) and bounded so that a tie-free
# dataset floors out near zero instead of running eta to -inf, and a tie-heavy
# one cannot blow up. These bounds never bind on realistic data.
_NU_MIN = 1e-3
_NU_MAX = 30.0

Outcome = Literal["A_WIN", "B_WIN", "TIE"]
Status = Literal["preliminary", "usable", "established"]


class BattleOutcome(BaseModel):
    """One settled battle: model A vs model B, and who won.

    ``A_WIN``/``B_WIN``/``TIE`` is stored raw and never collapsed, matching
    ``arena.votes.outcome``. ``model_a``/``model_b`` are opaque model ids.
    """

    model_a: str
    model_b: str
    outcome: Outcome


class ModelRating(BaseModel):
    """Rating for one model — the shape of an ``arena.leaderboard_snapshots`` row."""

    model_id: str
    rating_bt: float
    rating_elo: float
    # None when no CI is available (e.g. bootstrap_rounds=0, or the model never
    # appeared in a resample) — never NaN, which is invalid JSON and a silent
    # poison value in a NUMERIC column.
    ci_low: float | None
    ci_high: float | None
    ci_half_width: float | None = Field(default=None, ge=0)
    votes_total: int = Field(ge=0)
    wins: float = Field(ge=0)
    losses: float = Field(ge=0)
    ties: float = Field(ge=0)
    status: Status


class RatingResult(BaseModel):
    """Full leaderboard fit: the tie parameter plus one entry per model."""

    methodology_version: Literal["davidson-bt-001"] = METHODOLOGY_VERSION
    tie_param: float
    models: list[ModelRating]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Aggregate:
    """Sufficient statistics: unique models plus per-pair win/tie counts."""

    models: list[str]
    # Parallel arrays over unordered pairs (i < j):
    idx_i: npt.NDArray[np.int64]
    idx_j: npt.NDArray[np.int64]
    wins_ij: npt.NDArray[np.float64]  # times model i beat model j
    wins_ji: npt.NDArray[np.float64]  # times model j beat model i
    ties: npt.NDArray[np.float64]  # draws between i and j


def _aggregate(outcomes: Sequence[BattleOutcome]) -> _Aggregate:
    """Collapse raw battles into per-pair win/tie counts."""
    models = sorted({o.model_a for o in outcomes} | {o.model_b for o in outcomes})
    index = {m: k for k, m in enumerate(models)}

    # Map an unordered pair (lo, hi) -> [wins_lo_over_hi, wins_hi_over_lo, ties].
    pairs: dict[tuple[int, int], list[float]] = {}
    for o in outcomes:
        if o.model_a == o.model_b:
            continue
        a, b = index[o.model_a], index[o.model_b]
        lo, hi = (a, b) if a < b else (b, a)
        cell = pairs.setdefault((lo, hi), [0.0, 0.0, 0.0])
        if o.outcome == "TIE":
            cell[2] += 1.0
        else:
            winner = a if o.outcome == "A_WIN" else b
            cell[0 if winner == lo else 1] += 1.0

    keys = list(pairs.keys())
    idx_i = np.array([k[0] for k in keys], dtype=np.int64)
    idx_j = np.array([k[1] for k in keys], dtype=np.int64)
    wins_ij = np.array([pairs[k][0] for k in keys], dtype=np.float64)
    wins_ji = np.array([pairs[k][1] for k in keys], dtype=np.float64)
    ties = np.array([pairs[k][2] for k in keys], dtype=np.float64)
    return _Aggregate(
        models=models,
        idx_i=idx_i,
        idx_j=idx_j,
        wins_ij=wins_ij,
        wins_ji=wins_ji,
        ties=ties,
    )


# ---------------------------------------------------------------------------
# Davidson maximum-likelihood fit
# ---------------------------------------------------------------------------


def _fit(agg: _Aggregate, reg: float) -> tuple[npt.NDArray[np.float64], float]:
    """Fit Davidson theta (centered, mean 0) and tie parameter nu by MLE."""
    n = len(agg.models)
    if n == 0:
        return np.zeros(0, dtype=np.float64), _NU_MIN

    i, j = agg.idx_i, agg.idx_j
    w_ij, w_ji, tie = agg.wins_ij, agg.wins_ji, agg.ties
    n_pair = w_ij + w_ji + tie

    def neg_ll_grad(x: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
        theta = x[:n]
        nu = math.exp(x[n])
        ti, tj = theta[i], theta[j]
        ei, ej = np.exp(ti), np.exp(tj)
        eij = np.exp((ti + tj) / 2.0)
        denom = ei + ej + nu * eij
        log_d = np.log(denom)

        ll = float(
            np.sum(
                w_ij * (ti - log_d)
                + w_ji * (tj - log_d)
                + tie * (math.log(nu) + (ti + tj) / 2.0 - log_d)
            )
        )
        nll = -ll + reg * float(np.sum(theta * theta))

        p_i = ei / denom
        p_j = ej / denom
        p_tie = (nu * eij) / denom
        # d(LL)/d(theta) = observed - expected of the score (wins + 0.5 * ties).
        obs_i = w_ij + 0.5 * tie
        obs_j = w_ji + 0.5 * tie
        exp_i = n_pair * (p_i + 0.5 * p_tie)
        exp_j = n_pair * (p_j + 0.5 * p_tie)
        grad_theta = np.zeros(n, dtype=np.float64)
        np.add.at(grad_theta, i, -(obs_i - exp_i))
        np.add.at(grad_theta, j, -(obs_j - exp_j))
        grad_theta += 2.0 * reg * theta
        # d(LL)/d(eta) = observed ties - expected ties (chain rule on nu = e^eta).
        grad_eta = -float(np.sum(tie - n_pair * p_tie))

        grad = np.empty(n + 1, dtype=np.float64)
        grad[:n] = grad_theta
        grad[n] = grad_eta
        return nll, grad

    x0 = np.zeros(n + 1, dtype=np.float64)  # theta = 0, eta = 0 (nu = 1)
    bounds = [(None, None)] * n + [(math.log(_NU_MIN), math.log(_NU_MAX))]
    res = minimize(neg_ll_grad, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        # The ridge makes the objective strictly convex, so a healthy fit
        # converges easily; failure signals degenerate data (e.g. a disconnected
        # comparison graph), and res.x is not the MLE.
        raise ConvergenceError(
            f"Davidson MLE did not converge: {res.message!r} ({n} models, {len(agg.idx_i)} pairs)"
        )

    x_opt = np.asarray(res.x, dtype=np.float64)
    theta = x_opt[:n]
    theta = theta - float(np.mean(theta))  # recenter: theta is shift-invariant
    nu = float(math.exp(x_opt[n]))
    return theta, nu


def _elo(theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map centered log-strengths to the Elo scale."""
    return _ELO_BASE + _ELO_SCALE * theta


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _bootstrap(
    outcomes: Sequence[BattleOutcome],
    models: list[str],
    *,
    rounds: int,
    reg: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (ci_low, ci_high) Elo arrays aligned to ``models``.

    Resamples battles with replacement, refits, and aligns each replicate's Elo
    back to the canonical model order. Models absent from a replicate contribute
    no sample for that round (dropped, not imputed). A resample whose fit fails to
    converge is skipped (a few dropped replicates barely move the percentiles).
    """
    n = len(models)
    index = {m: k for k, m in enumerate(models)}
    arr = list(outcomes)
    n_battles = len(arr)
    # samples[k] collects every replicate Elo seen for model k.
    samples: list[list[float]] = [[] for _ in range(n)]

    failures = 0
    for _ in range(rounds):
        picks = rng.integers(0, n_battles, size=n_battles)
        resampled = [arr[p] for p in picks]
        agg = _aggregate(resampled)
        try:
            theta, _ = _fit(agg, reg)
        except ConvergenceError:
            failures += 1
            continue
        elo = _elo(theta)
        for local_k, model_id in enumerate(agg.models):
            samples[index[model_id]].append(float(elo[local_k]))

    if failures:
        logger.warning(
            "bootstrap: %d/%d resamples failed to converge and were skipped",
            failures,
            rounds,
        )

    ci_low = np.full(n, np.nan, dtype=np.float64)
    ci_high = np.full(n, np.nan, dtype=np.float64)
    for k in range(n):
        vals = np.array(samples[k], dtype=np.float64)
        # Fewer than two samples, or a zero-variance pool (degenerate/separated
        # data that refits identically every resample), is not a real interval.
        if vals.size < 2 or vals.min() == vals.max():
            continue
        ci_low[k] = float(np.percentile(vals, 2.5))
        ci_high[k] = float(np.percentile(vals, 97.5))
    return ci_low, ci_high


# ---------------------------------------------------------------------------
# Status classifier
# ---------------------------------------------------------------------------

# Provisional confidence tiers (methodology doc not yet on this branch). Driven
# by the bootstrap CI half-width in Elo points. ``preliminary`` is the floor: a
# model with no CI or a wide one reads as preliminary, not better. Tune once the
# methodology §7 thresholds land.
_CI_ESTABLISHED = 30.0
_CI_USABLE = 60.0


def classify_status(ci_half_width: float | None) -> Status:
    """Bucket a model into a confidence tier from its bootstrap CI half-width."""
    if ci_half_width is None or math.isnan(ci_half_width):
        return "preliminary"
    if ci_half_width <= _CI_ESTABLISHED:
        return "established"
    if ci_half_width <= _CI_USABLE:
        return "usable"
    return "preliminary"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_ratings(
    outcomes: Sequence[BattleOutcome],
    *,
    bootstrap_rounds: int = 1000,
    seed: int = 0,
    reg: float = 0.1,
) -> RatingResult:
    """Fit the Davidson-BT leaderboard from raw battle outcomes.

    Returns one :class:`ModelRating` per model seen, sorted by Elo descending.
    ``bootstrap_rounds`` controls CI precision/cost; ``seed`` makes the bootstrap
    reproducible; ``reg`` is the L2 ridge on theta.

    Limitation — disconnected comparison graph: if some models never share a
    direct or transitive matchup, their relative Elo is not identified by the
    data. The ridge keeps the fit finite and converging, but pins each component
    near 0, so cross-component ordering is arbitrary; ranks within a connected
    component stay valid, and the wide bootstrap CIs on poorly-connected models
    flag the low confidence. The intended remedy is upstream adaptive pairing
    (steer new battles to bridge components), not a hard failure here.
    """
    if bootstrap_rounds < 0:
        raise ValueError("bootstrap_rounds must be >= 0")
    if not math.isfinite(reg) or reg <= 0.0:
        raise ValueError("reg must be finite and > 0")

    agg = _aggregate(outcomes)
    n = len(agg.models)
    if n == 0:
        return RatingResult(tie_param=0.0, models=[])

    theta, nu = _fit(agg, reg)
    elo = _elo(theta)

    rng = np.random.default_rng(seed)
    ci_low, ci_high = _bootstrap(outcomes, agg.models, rounds=bootstrap_rounds, reg=reg, rng=rng)

    # Per-model win/loss/tie tallies from the per-pair counts.
    wins = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    ties = np.zeros(n, dtype=np.float64)
    np.add.at(wins, agg.idx_i, agg.wins_ij)
    np.add.at(wins, agg.idx_j, agg.wins_ji)
    np.add.at(losses, agg.idx_i, agg.wins_ji)
    np.add.at(losses, agg.idx_j, agg.wins_ij)
    np.add.at(ties, agg.idx_i, agg.ties)
    np.add.at(ties, agg.idx_j, agg.ties)

    entries: list[ModelRating] = []
    for k, model_id in enumerate(agg.models):
        votes_total = int(round(wins[k] + losses[k] + ties[k]))
        low = None if math.isnan(ci_low[k]) else float(ci_low[k])
        high = None if math.isnan(ci_high[k]) else float(ci_high[k])
        half_width = (high - low) / 2.0 if low is not None and high is not None else None
        entries.append(
            ModelRating(
                model_id=model_id,
                rating_bt=float(theta[k]),
                rating_elo=float(elo[k]),
                ci_low=low,
                ci_high=high,
                ci_half_width=half_width,
                votes_total=votes_total,
                wins=float(wins[k]),
                losses=float(losses[k]),
                ties=float(ties[k]),
                status=classify_status(half_width),
            )
        )

    entries.sort(key=lambda e: e.rating_elo, reverse=True)
    return RatingResult(tie_param=nu, models=entries)
