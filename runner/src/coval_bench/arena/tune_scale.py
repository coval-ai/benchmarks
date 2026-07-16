# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline simulation to tune the pairing ``SCALE`` constant.

NOT in the live pairing path — a research tool. We invent ground-truth model
strengths, then for each candidate ``SCALE`` run a prequential ("predict then
observe") simulation: at every step the *current* fitted leaderboard predicts the
next battle's outcome, we score that prediction, then we reveal the true outcome
and fold it into the data. A SCALE that picks more informative matchups converges
the fit faster, so its predictions are sharper sooner and its cumulative loss is
lower.

Loss is penalized cumulative cross-entropy::

    L = sum_i ( CE_i + K * max(0, CE_i - ln2) )
    CE_i = -ln(p)   p = predicted prob the fit assigned to the ACTUAL winner

``ln2`` is the coin-flip baseline; the K-term punishes confidently-wrong calls
(CE > ln2) extra. K=1 here (a confidently-wrong call costs ~2x a correct one).

The generative and predictive probabilities use the exact Davidson form that
``rating.compute_ratings`` fits, so "truth" and "model" never silently disagree.
Only *decisive* battles enter the loss, scored by the conditional win
probability p_i/(p_i+p_j); that is what keeps ln2 the true coin-flip baseline
(see DECISIONS note in the review summary).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from coval_bench.arena.pairing import SCALE, active_tts_models, select_pair
from coval_bench.arena.rating import (
    _ELO_SCALE,
    BattleOutcome,
    ConvergenceError,
    compute_ratings,
)
from coval_bench.db.models import PairingRating
from coval_bench.registries.models import RegisteredModel

_EPS = 1e-12

DEFAULT_SCALES = (100.0, 150.0, 200.0, 250.0, 300.0, 400.0)


@dataclass(frozen=True)
class ScaleResult:
    """Mean penalized loss for one candidate SCALE across replications."""

    scale: float
    loss: float  # mean over replications of L / n_decisive (per-battle, comparable)
    decisive: float  # mean count of decisive (loss-scored) battles


def _davidson(theta_i: float, theta_j: float, nu: float) -> tuple[float, float, float]:
    """Davidson (p_a_win, p_b_win, p_tie) — identical to the fitted likelihood."""
    ei = math.exp(theta_i)
    ej = math.exp(theta_j)
    eij = math.exp((theta_i + theta_j) / 2.0)
    denom = ei + ej + nu * eij
    return ei / denom, ej / denom, nu * eij / denom


def _penalized_ce(p_winner: float, k: float) -> float:
    """Penalized CE for one decisive battle: ``CE + k*(CE-ln2)`` when ``CE > ln2``.

    ``p_winner`` is the conditional win prob ``p_i/(p_i+p_j)``, so ``p=0.5`` is the
    ``ln2`` coin-flip baseline; the K-term only bites confidently-wrong calls.
    """
    ce = -math.log(max(p_winner, _EPS))
    return ce + k * max(0.0, ce - math.log(2.0))


def _model_id(m: RegisteredModel) -> str:
    return f"{m.provider}|{m.model}"


def _true_strengths(
    roster: Sequence[RegisteredModel], *, elo_spread: float, rng: random.Random
) -> dict[str, float]:
    """Assign each model a fixed true theta, evenly spread then shuffled.

    Even spacing (not random draws) guarantees a real tier structure to detect;
    the spread is given in Elo for interpretability and converted with the same
    constant ``_elo`` uses, so ``elo_spread`` is the true top-to-bottom gap.
    """
    n = len(roster)
    theta_spread = elo_spread / _ELO_SCALE
    grid = [(-0.5 + i / (n - 1)) * theta_spread for i in range(n)] if n > 1 else [0.0]
    rng.shuffle(grid)
    return {_model_id(m): theta for m, theta in zip(roster, grid, strict=True)}


def _sample_outcome(p_a: float, p_b: float, rng: random.Random) -> str:
    r = rng.random()
    if r < p_a:
        return "A_WIN"
    if r < p_a + p_b:
        return "B_WIN"
    return "TIE"


def _simulate_once(
    roster: Sequence[RegisteredModel],
    true_theta: dict[str, float],
    *,
    scale: float,
    n_battles: int,
    refit_every: int,
    bootstrap_rounds: int,
    true_nu: float,
    k: float,
    rng: random.Random,
    fit_seed: int,
) -> tuple[float, int]:
    """Run one prequential pass; return (total penalized loss, decisive count)."""
    outcomes: list[BattleOutcome] = []
    ratings: dict[tuple[str, str], PairingRating] = {}  # empty -> uniform pairing
    theta_fit: dict[str, float] = {}
    nu_fit = 1.0
    loss = 0.0
    decisive = 0

    for step in range(n_battles):
        a, b = select_pair(roster, ratings, scale=scale, rng=rng)
        id_a, id_b = _model_id(a), _model_id(b)

        # Reveal the true outcome.
        pa_t, pb_t, _ = _davidson(true_theta[id_a], true_theta[id_b], true_nu)
        outcome = _sample_outcome(pa_t, pb_t, rng)
        outcomes.append(BattleOutcome(model_a=id_a, model_b=id_b, outcome=outcome))

        # Score every decisive battle by conditional win prob (ln2 = coin flip).
        # Models not yet fitted use the neutral prior theta=0 (scores at ln2)
        # instead of being dropped, so the loss covers the full run.
        if outcome != "TIE":
            pa, pb, _ = _davidson(theta_fit.get(id_a, 0.0), theta_fit.get(id_b, 0.0), nu_fit)
            denom = pa + pb
            p_winner = (pa if outcome == "A_WIN" else pb) / denom if denom > 0 else 0.5
            loss += _penalized_ce(p_winner, k)
            decisive += 1

        if (step + 1) % refit_every == 0:
            try:
                result = compute_ratings(outcomes, bootstrap_rounds=bootstrap_rounds, seed=fit_seed)
            except ConvergenceError:
                continue  # keep the previous fit; data still too sparse/disconnected
            theta_fit = {mr.model_id: mr.rating_bt for mr in result.models}
            nu_fit = result.tie_param
            ratings = {}
            for mr in result.models:
                provider, model = mr.model_id.split("|", 1)
                ratings[provider, model] = PairingRating(
                    rating_elo=mr.rating_elo, ci_half_width=mr.ci_half_width
                )
    return loss, decisive


def tune_scale(
    *,
    scales: Sequence[float] = DEFAULT_SCALES,
    n_battles: int = 2000,
    refit_every: int = 100,
    replications: int = 3,
    bootstrap_rounds: int = 100,
    elo_spread: float = 800.0,
    true_nu: float = 1.0,
    k: float = 1.0,
    seed: int = 0,
) -> list[ScaleResult]:
    """Sweep candidate SCALEs; return per-SCALE mean per-battle penalized loss.

    The same true strengths are reused across all SCALEs within a replication
    (paired comparison: SCALE is the only thing that varies), and a fresh truth
    is drawn per replication. Loss is normalized by decisive count so SCALEs that
    happen to produce different tie rates stay comparable.
    """
    if not scales:
        raise ValueError("scales must contain at least one value")
    if n_battles <= 0:
        raise ValueError("n_battles must be > 0")
    if refit_every <= 0:
        raise ValueError("refit_every must be > 0")
    if replications <= 0:
        raise ValueError("replications must be > 0")
    if bootstrap_rounds < 0:
        raise ValueError("bootstrap_rounds must be >= 0")
    roster = active_tts_models()
    if len(roster) < 2:
        raise ValueError("need at least two active TTS models to simulate")

    results: list[ScaleResult] = []
    for scale in scales:
        per_rep_loss: list[float] = []
        per_rep_decisive: list[int] = []
        for rep in range(replications):
            # Three streams from (seed, rep): identical across scales (paired
            # comparison), distinct from each other. truth/sim share an engine so
            # are seeded by name; fit_seed feeds numpy (a different engine).
            truth_rng = random.Random(f"{seed}:{rep}:truth")  # noqa: S311
            true_theta = _true_strengths(roster, elo_spread=elo_spread, rng=truth_rng)
            sim_rng = random.Random(f"{seed}:{rep}:sim")  # noqa: S311
            loss, decisive = _simulate_once(
                roster,
                true_theta,
                scale=scale,
                n_battles=n_battles,
                refit_every=refit_every,
                bootstrap_rounds=bootstrap_rounds,
                true_nu=true_nu,
                k=k,
                rng=sim_rng,
                fit_seed=seed * 1000 + rep,
            )
            per_rep_loss.append(loss / decisive if decisive else math.inf)
            per_rep_decisive.append(decisive)
        results.append(
            ScaleResult(
                scale=scale,
                loss=sum(per_rep_loss) / len(per_rep_loss),
                decisive=sum(per_rep_decisive) / len(per_rep_decisive),
            )
        )
    return results


def render_loss_curve(results: Sequence[ScaleResult]) -> str:
    """Self-contained HTML table of the SCALE sweep (no chart, no shared helper)."""
    best = min(results, key=lambda r: r.loss)
    ordered = sorted(results, key=lambda r: r.scale)
    rows = "".join(
        f'<tr><th style="text-align:right">{r.scale:g}</th><td>{r.loss:.4f}</td>'
        f"<td>{'&larr; best' if r is best else ''}</td></tr>"
        for r in ordered
    )
    style = "body{font:13px/1.4 system-ui,sans-serif;margin:24px}td,th{padding:3px 8px}"
    return (
        "<!doctype html><meta charset=utf-8><title>Arena tune-scale</title>"
        f"<style>{style}</style><h1>Arena tune-scale</h1>"
        "<table><tr><th>SCALE</th><th>loss</th><th></th></tr>"
        f"{rows}</table><p>Recommended SCALE: <b>{best.scale:g}</b> "
        f"(committed default is {SCALE:g}).</p>"
    )
