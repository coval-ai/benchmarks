# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-turn LLM timing persistence tests."""

from __future__ import annotations

import asyncio
from typing import Any

import psycopg
import psycopg.errors
import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db.llm_turns import fetch_conversation_ttft, insert_turn

from .conftest import apply_migrations, open_pool

llm_pg = postgresql("pg_proc")


def test_turn_timings_average_the_latest_row_per_turn(
    llm_pg: psycopg.Connection[Any],
) -> None:
    apply_migrations(llm_pg)

    async def scenario() -> None:
        pool = await open_pool(llm_pg)
        try:
            assert await fetch_conversation_ttft(pool, []) == {}
            for simulation_id, turn_index, ttft_ms in (
                ("sim-1", 0, 100.0),
                ("sim-1", 1, 900.0),
                ("sim-1", 1, 300.0),
                ("sim-2", 0, 500.0),
            ):
                await insert_turn(
                    pool,
                    simulation_id=simulation_id,
                    turn_index=turn_index,
                    provider="phonely",
                    model="phonely-agent",
                    ttft_ms=ttft_ms,
                    total_ms=ttft_ms + 50,
                )
            assert await fetch_conversation_ttft(pool, ["sim-1", "sim-2", "missing"]) == {
                "sim-1": pytest.approx(0.2),
                "sim-2": pytest.approx(0.5),
            }
        finally:
            await pool.close()

    asyncio.run(scenario())


def test_turn_timing_constraints_reject_invalid_rows(llm_pg: psycopg.Connection[Any]) -> None:
    apply_migrations(llm_pg)
    with pytest.raises(psycopg.errors.CheckViolation), llm_pg.transaction():
        llm_pg.execute(
            """
            INSERT INTO benchmarks_v2.llm_turns
                (simulation_id, turn_index, provider, model, ttft_ms, total_ms)
            VALUES ('sim-1', 0, 'phonely', 'phonely-agent', 200, 100)
            """
        )
