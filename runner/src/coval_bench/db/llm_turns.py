# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistence for per-turn LLM timing measured by the proxy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg.rows
from psycopg_pool import AsyncConnectionPool


async def insert_turn(
    pool: AsyncConnectionPool[Any],
    *,
    simulation_id: str,
    turn_index: int,
    provider: str,
    model: str,
    ttft_ms: float,
    total_ms: float,
    output_tokens: int | None = None,
) -> None:
    """Append one completed assistant turn; database failures propagate."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO benchmarks_v2.llm_turns
                (simulation_id, turn_index, provider, model, ttft_ms, total_ms, output_tokens)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (simulation_id, turn_index, provider, model, ttft_ms, total_ms, output_tokens),
        )


async def fetch_conversation_ttft(
    pool: AsyncConnectionPool[Any], simulation_ids: Sequence[str]
) -> dict[str, float]:
    """Return mean TTFT in seconds per conversation, counting a retried turn once."""
    if not simulation_ids:
        return {}
    async with (
        pool.connection() as conn,
        conn.cursor(row_factory=psycopg.rows.dict_row) as cursor,
    ):
        await cursor.execute(
            """
            SELECT simulation_id, AVG(ttft_ms) / 1000.0 AS ttft_seconds
            FROM (
                SELECT DISTINCT ON (simulation_id, turn_index) simulation_id, ttft_ms
                FROM benchmarks_v2.llm_turns
                WHERE simulation_id = ANY(%s)
                ORDER BY simulation_id, turn_index, created_at DESC, id DESC
            ) latest
            GROUP BY simulation_id
            """,
            (list(simulation_ids),),
        )
        rows = await cursor.fetchall()
    return {row["simulation_id"]: row["ttft_seconds"] for row in rows}
