# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed a few demo battles from the example prompts.

A dev fixture so a labeler has battles to vote on before real user prompts arrive.
Each call generates ``per_domain`` battles per domain, using uniform pairing and
real synthesis (so it needs the TTS provider keys, like the benchmark runner).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import structlog

from coval_bench.arena.generate import generate_battle
from coval_bench.arena.pairing import gender_for_next_battle, roster_for, select_pair
from coval_bench.arena.prompts import EXAMPLE_PROMPTS
from coval_bench.config import Settings
from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.models import Battle
from coval_bench.registries import RegisteredModel
from coval_bench.registries.models import Gender

logger: structlog.BoundLogger = structlog.get_logger(__name__)


async def seed_demo_battles(
    settings: Settings,
    store: ArenaStore,
    models: Sequence[RegisteredModel],
    *,
    per_domain: int = 1,
    rng: random.Random | None = None,
) -> list[Battle]:
    """Generate up to ``per_domain`` demo battles for each example domain.

    Returns the battles that were created. A side whose synthesis fails is skipped
    (logged), never inserted — so the result may be shorter than requested.
    """
    picker = rng if rng is not None else random.Random()  # noqa: S311
    created: list[Battle] = []
    # Counted locally rather than read back from the table, so a seed run
    # balances its own output without depending on what is already stored.
    counts: dict[Gender, int] = {}

    for domain, prompts in EXAMPLE_PROMPTS.items():
        for prompt in prompts[:per_domain]:
            gender = gender_for_next_battle(counts, rng=picker)
            pair = select_pair(roster_for(models, gender), {}, rng=picker)
            battle = await generate_battle(
                settings,
                store,
                prompt=prompt,
                domain=domain,
                pair=pair,
                gender=gender,
                rng=picker,
            )
            if battle is None:
                logger.warning("arena_seed_skipped", domain=domain, prompt=prompt)
                continue
            counts[gender] = counts.get(gender, 0) + 1
            created.append(battle)

    return created
