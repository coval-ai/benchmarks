# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The tool-call log.

Every call is appended to ``benchmarks_v2.mock_tool_calls``, which is how a
conversation's tool use is recovered afterwards: what the agent asked for, which
seed answered, and in what order. The ingest joins it on ``simulation_id``, with
the caller number and a time window as the fallback for platforms that drop
unknown SIP headers.

The write is best-effort by design. The response is the product and the row is
telemetry, so a database hiccup must not turn into a failed tool call: the agent
is mid-conversation, and a 500 here would be graded as the agent mishandling a
broken tool when in fact the tool was fine. A lost row costs one call's
provenance; a lost call costs the scenario.
"""

from __future__ import annotations

from typing import Any

import structlog
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

logger = structlog.get_logger("coval_bench.mocktools")

_INSERT_SQL = """
    INSERT INTO benchmarks_v2.mock_tool_calls
        (simulation_id, caller_number, tool, args, matched_seed, response, latency_ms)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

_COUNT_BY_SIMULATION_SQL = """
    SELECT count(*) AS calls FROM benchmarks_v2.mock_tool_calls WHERE simulation_id = %s
"""

_COUNT_BY_CALLER_SQL = """
    SELECT count(*) AS calls FROM benchmarks_v2.mock_tool_calls
    WHERE caller_number = %s AND created_at > now() - interval '15 minutes'
"""


async def count_calls(
    pool: AsyncConnectionPool[Any],
    *,
    simulation_id: str | None = None,
    caller_number: str | None = None,
) -> int:
    """How many tool calls this conversation has already logged; 0 when it cannot be told.

    Counts by simulation id, else by caller number over the last 15 minutes. Never raises.
    """
    if simulation_id:
        query, key = _COUNT_BY_SIMULATION_SQL, simulation_id
    elif caller_number:
        query, key = _COUNT_BY_CALLER_SQL, caller_number
    else:
        return 0
    try:
        async with pool.connection() as conn:
            cursor = await conn.execute(query, (key,))
            row = await cursor.fetchone()
    except Exception:
        logger.warning("mock_tool_calls_not_counted", simulation_id=simulation_id, exc_info=True)
        return 0
    return int(row["calls"]) if row else 0


async def record_call(
    pool: AsyncConnectionPool[Any],
    *,
    tool: str,
    args: dict[str, Any],
    response: dict[str, Any],
    latency_ms: float,
    matched_seed: str | None = None,
    simulation_id: str | None = None,
    caller_number: str | None = None,
) -> bool:
    """Append one call to the log. Returns whether the row landed.

    Never raises: see the module docstring on why a logging failure must not
    become a tool failure.
    """
    try:
        async with pool.connection() as conn:
            await conn.execute(
                _INSERT_SQL,
                (
                    simulation_id,
                    caller_number,
                    tool,
                    Jsonb(args),
                    matched_seed,
                    Jsonb(response),
                    latency_ms,
                ),
            )
    except Exception:
        logger.warning(
            "mock_tool_call_not_recorded",
            tool=tool,
            simulation_id=simulation_id,
            exc_info=True,
        )
        return False
    return True
