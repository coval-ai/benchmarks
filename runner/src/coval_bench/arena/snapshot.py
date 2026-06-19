# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rating snapshot job — refit ratings from raw votes and persist one board.

Pure orchestration: reads every battle and vote via :class:`ArenaStore`, feeds
them to the pure :func:`compute_ratings` engine, and writes the resulting board
back through the store. No SQL and no rating math live here.
"""

from __future__ import annotations

from coval_bench.arena.rating import BattleOutcome, RatingResult, compute_ratings
from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.models import LeaderboardSnapshot, SnapshotStatus


def _model_id(provider: str, model: str) -> str:
    return f"{provider}/{model}"


async def _refit_and_persist(
    store: ArenaStore,
    *,
    metric_name: str,
    domain: str,
    bootstrap_rounds: int,
    seed: int,
) -> RatingResult:
    battles = {battle.id: battle for battle in await store.list_battles(limit=None)}
    votes = await store.list_votes()

    outcomes: list[BattleOutcome] = []
    id_to_pm: dict[str, tuple[str, str]] = {}
    for vote in votes:
        battle = battles.get(vote.battle_id)
        if battle is None:
            continue
        a_id = _model_id(battle.provider_a, battle.model_a)
        b_id = _model_id(battle.provider_b, battle.model_b)
        id_to_pm[a_id] = (battle.provider_a, battle.model_a)
        id_to_pm[b_id] = (battle.provider_b, battle.model_b)
        outcomes.append(BattleOutcome(model_a=a_id, model_b=b_id, outcome=vote.outcome.value))

    result = compute_ratings(outcomes, bootstrap_rounds=bootstrap_rounds, seed=seed)

    rows = [
        LeaderboardSnapshot(
            metric_name=metric_name,
            methodology_version=result.methodology_version,
            domain=domain,
            provider=id_to_pm[entry.model_id][0],
            model=id_to_pm[entry.model_id][1],
            rating_elo=entry.rating_elo,
            rating_bt=entry.rating_bt,
            ci_low=entry.ci_low,
            ci_high=entry.ci_high,
            ci_half_width=entry.ci_half_width,
            votes_total=entry.votes_total,
            wins=entry.wins,
            losses=entry.losses,
            ties=entry.ties,
            status=SnapshotStatus(entry.status),
        )
        for entry in result.models
    ]
    await store.insert_snapshot_board(rows)
    return result


async def run_snapshot(
    store: ArenaStore,
    *,
    metric_name: str = "naturalness",
    domain: str = "all",
    bootstrap_rounds: int = 1000,
    seed: int = 0,
    force: bool = False,
) -> RatingResult | None:
    """Refit the Davidson-BT board from all votes and persist it.

    Holds a non-blocking advisory lock so two runs never compute at once: if
    another run already holds it, this one returns ``None`` (skipped) rather than
    queueing. ``force=True`` bypasses the lock for a deliberate manual run.
    ``seed`` keeps the bootstrap CI reproducible. The result is empty — and
    nothing is written — when there are no votes; models with no votes never
    appear and are never persisted.
    """
    if force:
        return await _refit_and_persist(
            store,
            metric_name=metric_name,
            domain=domain,
            bootstrap_rounds=bootstrap_rounds,
            seed=seed,
        )
    async with store.snapshot_lock() as acquired:
        if not acquired:
            return None
        return await _refit_and_persist(
            store,
            metric_name=metric_name,
            domain=domain,
            bootstrap_rounds=bootstrap_rounds,
            seed=seed,
        )
