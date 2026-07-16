# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Model pairing for arena battles.

Picks which two of the roster render an incoming prompt. Weights informative
matchups: near-ties (most signal per vote) and under-measured models (widest CI).
Falls back to uniform random at cold start, before any ratings exist.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence

from coval_bench.db.models import PairingRating
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import MODEL_REGISTRY, ModelStatus, RegisteredModel

# Volume knob (Elo points): the rating gap at which a matchup's sampling priority
# decays by 1/e. Tuned offline and committed — deliberately a constant, never an
# env var, so the value running in prod is always the value in source.
SCALE = 300.0

# Which leaderboard board pairing reads ratings from. A single global board for
# now; per-domain boards can replace these once a domain has enough votes.
PAIRING_METRIC = "naturalness"
PAIRING_DOMAIN = "all"


def active_tts_models() -> list[RegisteredModel]:
    """The arena roster: every ACTIVE, arena-enabled TTS model in the registry."""
    return [
        m
        for m in MODEL_REGISTRY
        if m.benchmark is Benchmark.TTS and m.status is ModelStatus.ACTIVE and m.arena_enabled
    ]


def select_pair(
    models: Sequence[RegisteredModel],
    ratings: Mapping[tuple[str, str], PairingRating],
    *,
    scale: float = SCALE,
    rng: random.Random | None = None,
) -> tuple[RegisteredModel, RegisteredModel]:
    """Pick two distinct models to battle, weighting informative matchups.

    ``score(i, j) = exp(-gap / scale) * (ci_i + ci_j)`` with ``gap = |elo_i -
    elo_j|``. A pair is *sampled* in proportion to its score (not argmax) so the
    battle graph stays connected; the ``exp`` term favors near-ties, the CI term
    favors under-measured models. A model absent from ``ratings`` is treated as
    max uncertainty at the mean rating, so new models are auto-prioritized.

    Placeholders (mean rating, max CI) are taken only over active-roster models
    that are rated, so a stale board still carrying paused/retired models does not
    skew them. The max-CI placeholder is floored at 1.0 so a brand-new model stays
    prioritized even when every rated model has fully converged.

    Falls back to uniform random when no active model is rated yet (cold start) or
    when every model is fully converged (all weights zero). ``rng`` is injectable
    for deterministic tests; ``random.sample``/``choices`` keep the two distinct.
    """
    if len(models) < 2:
        raise ValueError("need at least two models to form a battle")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a positive finite number")
    picker = rng if rng is not None else random.Random()  # noqa: S311
    pool = list(models)

    def _uniform() -> tuple[RegisteredModel, RegisteredModel]:
        first, second = picker.sample(pool, 2)
        return first, second

    rated = [r for m in pool if (r := ratings.get((m.provider, m.model))) is not None]
    if not rated:
        return _uniform()

    mean_elo = sum(r.rating_elo for r in rated) / len(rated)
    max_ci = max(
        (r.ci_half_width for r in rated if r.ci_half_width is not None),
        default=1.0,
    )
    max_ci = max(max_ci, 1.0)

    def elo(m: RegisteredModel) -> float:
        r = ratings.get((m.provider, m.model))
        return r.rating_elo if r is not None else mean_elo

    def ci(m: RegisteredModel) -> float:
        r = ratings.get((m.provider, m.model))
        if r is None or r.ci_half_width is None:
            return max_ci
        return r.ci_half_width

    pairs = list(itertools.combinations(pool, 2))
    weights = [math.exp(-abs(elo(a) - elo(b)) / scale) * (ci(a) + ci(b)) for a, b in pairs]
    if sum(weights) <= 0.0:
        return _uniform()
    first, second = picker.choices(pairs, weights=weights, k=1)[0]
    return first, second
