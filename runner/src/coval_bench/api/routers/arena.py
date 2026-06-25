# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice Arena endpoints.

``GET  /v1/arena/battle``       — a battle to vote on (blind: no model identities).
``GET  /v1/arena/battle/{id}``  — a specific battle.
``GET  /v1/arena/leaderboard``  — the latest computed board for a metric/domain.
``POST /v1/arena/vote``         — record a labeler's vote (labeler-only at MVP).

Reads hit the pool directly; the write path uses the arena DB-access layer
(``ArenaStore``). Leaderboard rows are produced by the snapshot job, so the
leaderboard is empty until that job has run.

The battle payload deliberately omits provider/model identities to keep voting
blind — the A/B -> model mapping stays server-side, used only when a vote is
recorded.
"""

from __future__ import annotations

import hmac
import uuid
from typing import Any

import psycopg.rows
import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import capture_api_event, get_pool, get_posthog, get_settings
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    ArenaLeaderboardResponse,
    BattleCreate,
    BattleOut,
    LeaderboardEntryOut,
    RevealModelOut,
    RevealOut,
    VoteIn,
    VoteOut,
)
from coval_bench.arena.generate import generate_battle
from coval_bench.arena.pairing import (
    PAIRING_DOMAIN,
    PAIRING_METRIC,
    active_tts_models,
    select_pair,
)
from coval_bench.config import Settings
from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.models import VoteOutcome, VoterType

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["arena"])


# Columns exposed by the (blind) battle endpoints — provider/model are withheld.
_BATTLE_COLS = "id, prompt_text, domain, audio_a_url, audio_b_url"

# Latest board for a (metric, domain): the rows of the single most recent
# (computed_at, methodology_version), so two methodology versions computed at the
# same timestamp never mix into one board.
_LEADERBOARD_SQL = """
    WITH latest AS (
        SELECT computed_at, methodology_version
        FROM arena.leaderboard_snapshots
        WHERE metric_name = %(metric)s AND domain = %(domain)s
        ORDER BY computed_at DESC, methodology_version DESC
        LIMIT 1
    )
    SELECT s.provider, s.model, s.methodology_version, s.computed_at,
           s.rating_elo, s.rating_bt, s.ci_low, s.ci_high, s.ci_half_width,
           s.votes_total, s.wins, s.losses, s.ties, s.status
    FROM arena.leaderboard_snapshots s
    JOIN latest l USING (computed_at, methodology_version)
    WHERE s.metric_name = %(metric)s AND s.domain = %(domain)s
    ORDER BY s.rating_elo DESC
"""


def _is_authenticated_labeler(provided: str | None, settings: Settings) -> bool:
    """True only if a labeler key is configured and the presented key matches it."""
    expected = settings.arena_labeler_key
    if expected is None or provided is None:
        return False
    return hmac.compare_digest(provided, expected.get_secret_value())


def require_labeler(
    x_labeler_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Gate a route behind the labeler key, raising 404 (not 403) so a locked arena is
    indistinguishable from a route that does not exist — keeping its existence
    confidential until launch."""
    if not _is_authenticated_labeler(x_labeler_key, settings):
        raise HTTPException(404)


@router.get("/arena/battle", response_model=BattleOut, dependencies=[Depends(require_labeler)])
@limiter.limit("60/minute")
async def get_battle(
    request: Request,  # required by slowapi
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
) -> BattleOut:
    """Return one battle to vote on, chosen at random. 404 if none exist."""
    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        # Placeholder selection: a uniformly random battle. Adaptive pairing will
        # replace this to surface the most informative matchups.
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


@router.get(
    "/arena/battle/{battle_id}",
    response_model=BattleOut,
    dependencies=[Depends(require_labeler)],
)
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


@router.get(
    "/arena/leaderboard",
    response_model=ArenaLeaderboardResponse,
    dependencies=[Depends(require_labeler)],
)
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


@router.post("/arena/vote", response_model=VoteOut, status_code=201)
@limiter.limit("60/minute")
async def cast_vote(
    request: Request,  # required by slowapi
    vote: VoteIn,
    x_labeler_key: str | None = Header(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    settings: Settings = Depends(get_settings),
    posthog_client: Posthog | None = Depends(get_posthog),
) -> VoteOut:
    """Record a labeler's vote on a battle (labeler-only at MVP; others get 403)."""
    if not _is_authenticated_labeler(x_labeler_key, settings):
        raise HTTPException(403, "external voting is not enabled")
    store = ArenaStore(pool)
    if await store.get_battle(vote.battle_id) is None:
        raise HTTPException(404, f"battle {vote.battle_id} not found")
    recorded = await store.upsert_vote(
        battle_id=vote.battle_id,
        outcome=VoteOutcome(vote.outcome),
        voter_type=VoterType.LABELER,
        voter_id=vote.voter_id,
    )
    capture_api_event(
        posthog_client,
        "arena_vote_cast",
        {
            "outcome": vote.outcome,
            "voter_type": VoterType.LABELER.value,
            "$process_person_profile": False,
        },
    )
    return VoteOut.model_validate(recorded.model_dump())


@router.post(
    "/arena/battle",
    response_model=BattleOut,
    status_code=201,
    dependencies=[Depends(require_labeler)],
)
@limiter.limit("60/minute")
async def create_battle(
    request: Request,  # required by slowapi
    body: BattleCreate,
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    settings: Settings = Depends(get_settings),
    posthog_client: Posthog | None = Depends(get_posthog),
) -> BattleOut:
    """Generate a battle from a prompt: pick two models, synthesize, persist (blind).

    Pairing weights informative matchups from the latest ratings, falling back to
    uniform at cold start. Returns 502 if either side fails to synthesize (no battle
    is written). Guarded by a global daily cap (``arena_daily_battle_cap``, ``<= 0``
    disables), checked before synthesis so a tripped cap spends nothing (429). Known
    limits: it counts successful battles only (a failed one still costs ~2 calls,
    uncounted but logged), and the non-transactional check may overshoot slightly
    under concurrency (no data is overwritten).
    """
    store = ArenaStore(pool)
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(422, "prompt must not be empty")

    cap = settings.arena_daily_battle_cap
    if cap > 0 and await store.count_battles_today() >= cap:
        capture_api_event(
            posthog_client,
            "arena_battle_cap_reached",
            {"$process_person_profile": False},
        )
        raise HTTPException(429, "daily generation limit reached")

    models = active_tts_models()
    if len(models) < 2:
        raise HTTPException(503, "not enough active TTS models to form a battle")
    ratings = await store.get_latest_ratings(metric_name=PAIRING_METRIC, domain=PAIRING_DOMAIN)
    pair = select_pair(models, ratings)
    battle = await generate_battle(
        settings,
        store,
        prompt=prompt,
        domain=body.domain,
        pair=pair,
    )
    if battle is None:
        raise HTTPException(502, "audio synthesis failed for one or both models")

    capture_api_event(
        posthog_client,
        "arena_battle_generated",
        {"$process_person_profile": False},
    )
    return BattleOut.model_validate(battle.model_dump())


@router.get("/arena/battle/{battle_id}/reveal", response_model=RevealOut)
@limiter.limit("60/minute")
async def reveal_battle(
    request: Request,  # required by slowapi
    battle_id: uuid.UUID,
    x_labeler_key: str | None = Header(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    settings: Settings = Depends(get_settings),
) -> RevealOut:
    """De-anonymize a battle's two sides. Labeler-only, and only after a vote exists."""
    if not _is_authenticated_labeler(x_labeler_key, settings):
        raise HTTPException(403, "reveal is not enabled")
    store = ArenaStore(pool)
    battle = await store.get_battle(battle_id)
    if battle is None:
        raise HTTPException(404, f"battle {battle_id} not found")
    if not await store.list_votes(battle_id=battle_id, limit=1):
        raise HTTPException(409, "battle has not been voted on yet")
    return RevealOut(
        a=RevealModelOut(provider=battle.provider_a, model=battle.model_a, label=battle.model_a),
        b=RevealModelOut(provider=battle.provider_b, model=battle.model_b, label=battle.model_b),
    )
