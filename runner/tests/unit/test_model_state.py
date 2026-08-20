# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.db.model_state and the 20260820_0019 migration seed.

Uses ``pytest-postgresql`` (embedded ``pg_ctl``, no Docker) with the real
Alembic migrations, so the seed under test is exactly what ships.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql

from coval_bench.db.model_state import (
    DEFAULT_STATE,
    assume_all_active,
    fetch_model_states,
    fetch_recent_history,
    registry_keys,
    set_model_state,
)
from coval_bench.registries import MODEL_REGISTRY, Benchmark
from tests.unit.conftest import apply_migrations, open_pool

state_pg = postgresql("pg_proc")  # shared server from conftest, own per-test DB


@pytest_asyncio.fixture
async def pool(state_pg: Any) -> Any:
    apply_migrations(state_pg)
    p: AsyncConnectionPool[Any] = await open_pool(state_pg)
    yield p
    await p.close()


_NOVA3 = (Benchmark.STT, "deepgram", "nova-3")


async def test_seed_covers_every_registry_model(pool: AsyncConnectionPool[Any]) -> None:
    """Every current registry entry has a seeded row (no missing-row defaults)."""
    states = await fetch_model_states(pool)
    assert set(states) == set(registry_keys())
    assert all(s.updated_by == "seed:20260820_0019" for s in states.values())


async def test_seed_reproduces_the_status_snapshot(pool: AsyncConnectionPool[Any]) -> None:
    """The seed maps the old statuses onto the booleans as designed.

    Spot checks per old status: ACTIVE -> running+shown, PAUSED -> shown only,
    EARLY_ACCESS -> running only, RETIRED/PENDING -> neither; plus the totals
    of the 2026-08-20 snapshot (51/7/11/24).
    """
    states = await fetch_model_states(pool)

    def rs(key: tuple[Benchmark, str, str]) -> tuple[bool, bool]:
        return states[key].running, states[key].shown

    assert rs(_NOVA3) == (True, True)  # was ACTIVE
    assert rs((Benchmark.STT, "azure", "default")) == (False, True)  # was PAUSED
    assert rs((Benchmark.TTS, "baseten", "qwen3-tts-1.7b")) == (True, False)  # was EARLY_ACCESS
    assert rs((Benchmark.TTS, "rime", "mistv2")) == (False, False)  # was RETIRED
    assert rs((Benchmark.STT, "soniox", "stt-rt-v4")) == (False, False)  # was PENDING

    combos = [(s.running, s.shown) for s in states.values()]
    assert combos.count((True, True)) == 51
    assert combos.count((False, True)) == 7
    assert combos.count((True, False)) == 11
    assert combos.count((False, False)) == 24


async def test_seed_wrote_history_rows(pool: AsyncConnectionPool[Any]) -> None:
    history = await fetch_recent_history(pool)
    assert set(history) == set(registry_keys())
    seeded = history[_NOVA3][0]
    assert (seeded.old_running, seeded.old_shown) == (None, None)
    assert (seeded.new_running, seeded.new_shown) == (True, True)


async def test_missing_row_defaults_to_hidden(pool: AsyncConnectionPool[Any]) -> None:
    """A registry entry with no row runs under embargo (never leaks, still runs)."""
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM benchmarks_v2.model_state"
            " WHERE benchmark = 'STT' AND provider = 'deepgram' AND model = 'nova-3'"
        )
    states = await fetch_model_states(pool)
    assert states[_NOVA3] == DEFAULT_STATE
    assert DEFAULT_STATE.running and not DEFAULT_STATE.shown


async def test_orphan_rows_are_dropped(pool: AsyncConnectionPool[Any]) -> None:
    """A row whose model left the registry is invisible to every reader."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO benchmarks_v2.model_state
                (benchmark, provider, model, running, shown, updated_by)
            VALUES ('STT', 'ghost', 'long-deleted', true, true, 'test')
            """
        )
    states = await fetch_model_states(pool)
    assert (Benchmark.STT, "ghost", "long-deleted") not in states
    history = await fetch_recent_history(pool)
    assert (Benchmark.STT, "ghost", "long-deleted") not in history


async def test_set_model_state_upserts_and_records_history(
    pool: AsyncConnectionPool[Any],
) -> None:
    state = await set_model_state(
        pool,
        benchmark=Benchmark.STT,
        provider="deepgram",
        model="nova-3",
        running=False,
        shown=True,
        changed_by="someone@coval.dev",
    )
    assert (state.running, state.shown) == (False, True)
    assert state.updated_by == "someone@coval.dev"
    assert state.updated_at is not None

    states = await fetch_model_states(pool)
    assert states[_NOVA3] == state

    changes = (await fetch_recent_history(pool))[_NOVA3]
    assert (changes[0].old_running, changes[0].old_shown) == (True, True)
    assert (changes[0].new_running, changes[0].new_shown) == (False, True)
    assert changes[0].changed_by == "someone@coval.dev"
    assert changes[0].changed_at == state.updated_at
    # The seed entry sits beneath it, newest first.
    assert (changes[1].old_running, changes[1].old_shown) == (None, None)


async def test_set_model_state_first_write_after_delete_records_null_old(
    pool: AsyncConnectionPool[Any],
) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM benchmarks_v2.model_state"
            " WHERE benchmark = 'STT' AND provider = 'deepgram' AND model = 'nova-3'"
        )
    await set_model_state(
        pool,
        benchmark=Benchmark.STT,
        provider="deepgram",
        model="nova-3",
        running=True,
        shown=True,
        changed_by="someone@coval.dev",
    )
    latest = (await fetch_recent_history(pool))[_NOVA3][0]
    assert (latest.old_running, latest.old_shown) == (None, None)


def test_assume_all_active_covers_the_registry() -> None:
    states = assume_all_active()
    assert set(states) == set(registry_keys())
    assert all(s.running and s.shown for s in states.values())
    assert len(states) == len(MODEL_REGISTRY)


@pytest.mark.parametrize(
    ("running", "shown"),
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_round_trip_every_state(
    pool: AsyncConnectionPool[Any], running: bool, shown: bool
) -> None:
    await set_model_state(
        pool,
        benchmark=Benchmark.TTS,
        provider="rime",
        model="coda",
        running=running,
        shown=shown,
        changed_by="someone@coval.dev",
    )
    states = await fetch_model_states(pool)
    key = (Benchmark.TTS, "rime", "coda")
    assert (states[key].running, states[key].shown) == (running, shown)


async def test_states_reproduce_the_old_consumer_sets(pool: AsyncConnectionPool[Any]) -> None:
    """The seeded booleans reproduce what each consumer derived from statuses.

    The pre-merge check from the design doc: orchestrator selection (running),
    embargo set (running and not shown), disabled set (neither), and the arena
    roster (running and shown and arena_enabled) all match the 2026-08-20
    registry snapshot.
    """
    from coval_bench.api.internal import embargoed_pairs
    from coval_bench.arena.pairing import active_tts_models
    from coval_bench.registries.models import Gender  # noqa: F401 — registry import guard

    states = await fetch_model_states(pool)

    scheduled = {key for key, s in states.items() if s.running}
    assert len(scheduled) == 51 + 11  # old ACTIVE + EARLY_ACCESS

    embargo = embargoed_pairs(states)
    assert ("baseten", "qwen3-tts-1.7b") in embargo
    assert ("xai", "grok-realtime") in embargo  # retired board key rides along
    assert ("deepgram", "nova-3") not in embargo

    disabled = {key for key, s in states.items() if not s.running and not s.shown}
    assert len(disabled) == 24  # old RETIRED + PENDING

    roster = active_tts_models(states)
    assert roster, "arena roster must not be empty"
    roster_keys = {(m.benchmark, m.provider, m.model) for m in roster}
    assert all(states[k].running and states[k].shown for k in roster_keys)
    assert all(m.arena_enabled for m in roster)
