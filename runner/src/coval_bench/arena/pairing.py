# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Model pairing for arena battles.

Uniform-random selection (cold-start strategy): pick two distinct models for an
incoming prompt. A later adaptive strategy will weight by rating uncertainty and
closeness (CI reduction); this function is the seam it slots into — the caller
stays the same, only the selection changes.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import MODEL_REGISTRY, ModelStatus, RegisteredModel


def active_tts_models() -> list[RegisteredModel]:
    """The arena roster: every ACTIVE TTS model in the registry."""
    return [
        m for m in MODEL_REGISTRY if m.benchmark is Benchmark.TTS and m.status is ModelStatus.ACTIVE
    ]


def select_pair(
    models: Sequence[RegisteredModel],
    *,
    rng: random.Random | None = None,
) -> tuple[RegisteredModel, RegisteredModel]:
    """Pick two distinct models to battle, uniformly at random.

    ``rng`` is injectable for deterministic tests. Raises if fewer than two
    models are available. ``random.sample`` guarantees the two are distinct, so a
    self-battle is impossible.
    """
    if len(models) < 2:
        raise ValueError("need at least two models to form a battle")
    picker = rng if rng is not None else random.Random()  # noqa: S311
    first, second = picker.sample(list(models), 2)
    return first, second
