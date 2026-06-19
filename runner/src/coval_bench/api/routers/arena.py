# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice Arena read endpoints.

``GET /v1/arena/battle``       — a battle to vote on (blind: no model identities).
``GET /v1/arena/battle/{id}``  — a specific battle.
``GET /v1/arena/leaderboard``  — the latest computed board for a metric/domain.

Reads only. Battles/votes are written elsewhere; leaderboard rows are produced by
the snapshot job, so the leaderboard is empty until that job has run. Queries hit
the pool directly, matching the other routers; this module does not depend on the
arena DB-access layer, so it can land independently.

The battle payload deliberately omits provider/model identities to keep voting
blind — the A/B -> model mapping stays server-side, used only when a vote is
recorded.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import psycopg.rows
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from starlette.requests import Request

from coval_bench.api.deps import capture_api_event, get_pool, get_posthog
from coval_bench.api.ratelimit import limiter

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["arena"])


class BattleOut(BaseModel):
    """A battle to vote on. Blind by design: no provider/model identities."""

    id: uuid.UUID
    prompt_text: str
    domain: str | None
    audio_a_url: str
    audio_b_url: str


class LeaderboardEntryOut(BaseModel):
    """One model's row within an arena leaderboard board."""

    provider: str
    model: str
    rating_elo: float
    rating_bt: float
    ci_low: float | None
    ci_high: float | None
    ci_half_width: float | None
    votes_total: int
    wins: float
    losses: float
    ties: float
    status: str


class ArenaLeaderboardResponse(BaseModel):
    """The latest board for a metric/domain, or empty if none computed yet."""

    metric: str
    domain: str
    computed_at: datetime | None
    methodology_version: str | None
    entries: list[LeaderboardEntryOut]


# Columns exposed by the (blind) battle endpoints — provider/model are withheld.
_BATTLE_COLS = "id, prompt_text, domain, audio_a_url, audio_b_url"

# Latest board for a (metric, domain): the rows sharing the max computed_at.
_LEADERBOARD_SQL = """
    SELECT provider, model, methodology_version, computed_at,
           rating_elo, rating_bt, ci_low, ci_high, ci_half_width,
           votes_total, wins, losses, ties, status
    FROM arena.leaderboard_snapshots
    WHERE metric_name = %(metric)s
      AND domain = %(domain)s
      AND computed_at = (
          SELECT max(computed_at) FROM arena.leaderboard_snapshots
          WHERE metric_name = %(metric)s AND domain = %(domain)s
      )
    ORDER BY rating_elo DESC
"""


@router.get("/arena/battle", response_model=BattleOut)
@limiter.limit("60/minute")
async def get_battle(
    request: Request,  # required by slowapi
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
) -> BattleOut:
    """Return one battle to vote on, chosen at random. 404 if none exist."""
    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = await conn.execute(
            f"SELECT {_BATTLE_COLS} FROM arena.battles ORDER BY random() LIMIT 1"  # noqa: S608
        )
        row = await rows.fetchone()
    if row is None:
        raise HTTPException(404, "no battles available")
    capture_api_event(
        posthog_client,
        "arena_battle_served",
        {"$process_person_profile": False},
    )
    return BattleOut.model_validate(row)


@router.get("/arena/battle/{battle_id}", response_model=BattleOut)
@limiter.limit("60/minute")
async def get_battle_by_id(
    request: Request,  # required by slowapi
    battle_id: uuid.UUID,
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> BattleOut:
    """Return a specific battle by id. 404 if not found (422 if id is not a UUID)."""
    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = await conn.execute(
            f"SELECT {_BATTLE_COLS} FROM arena.battles WHERE id = %(id)s",  # noqa: S608
            {"id": battle_id},
        )
        row = await rows.fetchone()
    if row is None:
        raise HTTPException(404, f"battle {battle_id} not found")
    return BattleOut.model_validate(row)


@router.get("/arena/leaderboard", response_model=ArenaLeaderboardResponse)
@limiter.limit("60/minute")
async def get_arena_leaderboard(
    request: Request,  # required by slowapi
    metric: str = Query(default="naturalness"),
    domain: str = Query(default="all"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
) -> ArenaLeaderboardResponse:
    """Return the latest computed board for ``metric``/``domain`` (empty if none)."""
    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = await conn.execute(_LEADERBOARD_SQL, {"metric": metric, "domain": domain})
        board = await rows.fetchall()

    entries = [LeaderboardEntryOut.model_validate(r) for r in board]
    capture_api_event(
        posthog_client,
        "arena_leaderboard_queried",
        {
            "metric": metric,
            "domain": domain,
            "entry_count": len(entries),
            "$process_person_profile": False,
        },
    )
    return ArenaLeaderboardResponse(
        metric=metric,
        domain=domain,
        computed_at=board[0]["computed_at"] if board else None,
        methodology_version=board[0]["methodology_version"] if board else None,
        entries=entries,
    )
