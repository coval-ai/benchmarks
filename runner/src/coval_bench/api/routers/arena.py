# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice Arena endpoints.

``GET  /v1/arena/battle``          — a battle to vote on (blind: no model identities).
``GET  /v1/arena/battle/{id}``     — a specific battle.
``GET  /v1/arena/example-prompt``  — a random seed-bank prompt with its domain.
``GET  /v1/arena/leaderboard``     — the latest computed board for a metric/domain.
``GET  /v1/arena/admin/cooccurrence`` — who-battled-whom heatmap (labeler-only, HTML).
``GET  /v1/arena/admin/convergence``  — CI-vs-votes per model (labeler-only, HTML).
``POST /v1/arena/vote``            — record a labeler's vote (labeler-only at MVP).

Reads hit the pool directly; the write path uses the arena DB-access layer
(``ArenaStore``). Leaderboard rows are produced by the snapshot job, so the
leaderboard is empty until that job has run.

The battle payload deliberately omits provider/model identities to keep voting
blind — the A/B -> model mapping stays server-side, used only when a vote is
recorded.
"""

from __future__ import annotations

import asyncio
import hmac
import secrets
import uuid
from typing import Any

import psycopg.rows
import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from slowapi.util import get_remote_address
from starlette.requests import Request

from coval_bench.api.deps import capture_api_event, get_pool, get_posthog, get_settings
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    ArenaLeaderboardResponse,
    BattleCreate,
    BattleOut,
    ExamplePromptOut,
    LeaderboardEntryOut,
    RevealModelOut,
    RevealOut,
    VoteIn,
    VoteOut,
)
from coval_bench.arena.audio_store import clip_url
from coval_bench.arena.domains import ArenaDomain
from coval_bench.arena.generate import generate_battle
from coval_bench.arena.moderation import moderation_verdict, pii_match
from coval_bench.arena.monitoring import (
    build_convergence,
    build_cooccurrence,
    render_convergence,
    render_cooccurrence,
)
from coval_bench.arena.pairing import (
    PAIRING_DOMAIN,
    PAIRING_METRIC,
    gender_for_next_battle,
    roster_for,
    select_pair,
)
from coval_bench.arena.prompts import EXAMPLE_PROMPTS
from coval_bench.config import Settings
from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.models import VoteOutcome, VoterType
from coval_bench.registries import MODEL_REGISTRY, Benchmark, ModelStatus

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["arena"], include_in_schema=False)


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

# TTS models the site hides (retired/pending/early-access). Their votes still
# feed the rating fit, but boards computed before a retirement must not keep
# showing them.
_HIDDEN_TTS_MODELS = frozenset(
    (m.provider, m.model)
    for m in MODEL_REGISTRY
    if m.benchmark is Benchmark.TTS
    and m.status in (ModelStatus.RETIRED, ModelStatus.PENDING, ModelStatus.EARLY_ACCESS)
)


def _is_authenticated_labeler(provided: str | None, settings: Settings) -> bool:
    """True only if a labeler key is configured and the presented key matches it."""
    expected = settings.arena_labeler_key
    if expected is None or provided is None:
        return False
    return hmac.compare_digest(
        provided.encode("utf-8"), expected.get_secret_value().encode("utf-8")
    )


def require_labeler(
    x_labeler_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Gate a route behind the labeler key, raising 404 (not 403) so a locked arena is
    indistinguishable from a route that does not exist — keeping its existence
    confidential until launch."""
    if not _is_authenticated_labeler(x_labeler_key, settings):
        raise HTTPException(404)


def _client_key(request: Request) -> str:
    """Rate-limit key: the end client forwarded by the BFF, else the caller address.

    All browser traffic reaches this API through the BFF, so keying on the remote
    address would pool every labeler into one shared bucket."""
    return request.headers.get("x-arena-client") or get_remote_address(request)


async def _battle_out(settings: Settings, row: dict[str, Any]) -> BattleOut:
    """Resolve stored clip keys to fresh playable URLs (off the loop — GCS signing is I/O)."""
    row = dict(row)
    row["audio_a_url"], row["audio_b_url"] = await asyncio.gather(
        asyncio.to_thread(clip_url, settings, row["audio_a_url"]),
        asyncio.to_thread(clip_url, settings, row["audio_b_url"]),
    )
    return BattleOut.model_validate(row)


@router.get("/arena/battle", response_model=BattleOut, dependencies=[Depends(require_labeler)])
@limiter.limit("60/minute")
async def get_battle(
    request: Request,  # required by slowapi
    domain: ArenaDomain | None = Query(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    settings: Settings = Depends(get_settings),
    posthog_client: Posthog | None = Depends(get_posthog),
) -> BattleOut:
    """Return one battle to vote on, chosen at random (within ``domain`` when given).
    404 if none exist."""
    # Placeholder selection: a uniformly random battle. Adaptive pairing will
    # replace this to surface the most informative matchups. With GCS-backed
    # clips, skip battles older than the bucket retention — their audio is gone.
    # Only gendered battles. Their votes still count, but serving a legacy
    # cross-gender pairing would keep adding the very comparison this stopped
    # making — the old share should shrink from here, not grow.
    conditions = ["gender IS NOT NULL"]
    params: dict[str, Any] = {}
    if settings.arena_gcs_bucket:
        conditions.append("created_at >= now() - make_interval(days => %(days)s)")
        params["days"] = settings.arena_clip_retention_days
    if domain is not None:
        conditions.append("domain = %(domain)s")
        params["domain"] = domain
    where = f"WHERE {' AND '.join(conditions)} " if conditions else ""
    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = await conn.execute(
            f"SELECT {_BATTLE_COLS} FROM arena.battles {where}"  # noqa: S608
            "ORDER BY random() LIMIT 1",
            params,
        )
        row = await rows.fetchone()
    if row is None:
        raise HTTPException(404, "no battles available")
    capture_api_event(
        posthog_client,
        "arena_battle_served",
        {"$process_person_profile": False},
    )
    return await _battle_out(settings, row)


@router.get(
    "/arena/example-prompt",
    response_model=ExamplePromptOut,
    dependencies=[Depends(require_labeler)],
)
@limiter.limit("60/minute", key_func=_client_key)
async def get_example_prompt(
    request: Request,  # required by slowapi
    domain: ArenaDomain | None = Query(
        None,
        description="Draw only from this domain's bank. Omitted -> a random domain.",
    ),
) -> ExamplePromptOut:
    """Return a random prompt from the seed bank, tagged with the domain it came from.

    Callers that already know the domain pass it, so the prompt and the domain a battle
    is filed under always agree; an unfiltered draw would hand back a prompt from some
    other domain and mislabel every per-domain board built from it.
    """
    chosen = domain if domain is not None else secrets.choice(list(EXAMPLE_PROMPTS))
    return ExamplePromptOut(prompt=secrets.choice(EXAMPLE_PROMPTS[chosen]), domain=chosen)


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
    settings: Settings = Depends(get_settings),
) -> BattleOut:
    """Return a specific battle by id. 404 if not found (422 if id is not a UUID)."""
    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        if settings.arena_gcs_bucket:
            rows = await conn.execute(
                f"SELECT {_BATTLE_COLS} FROM arena.battles "  # noqa: S608
                "WHERE id = %(id)s "
                "AND created_at >= now() - make_interval(days => %(days)s)",
                {"id": battle_id, "days": settings.arena_clip_retention_days},
            )
        else:
            rows = await conn.execute(
                f"SELECT {_BATTLE_COLS} FROM arena.battles WHERE id = %(id)s",  # noqa: S608
                {"id": battle_id},
            )
        row = await rows.fetchone()
    if row is None:
        raise HTTPException(404, f"battle {battle_id} not found")
    return await _battle_out(settings, row)


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

    entries = [
        LeaderboardEntryOut.model_validate(r)
        for r in board
        if (r["provider"], r["model"]) not in _HIDDEN_TTS_MODELS
    ]
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
    is written).

    Prompts are screened before anything is spent — personal data locally, then content
    via the moderation API (422), and 503 if the moderator is unreachable while
    ``arena_moderation_fail_closed``. Screening precedes the cap so a rejected prompt
    costs neither money nor quota.

    Then a global daily cap (``arena_daily_battle_cap``, ``<= 0`` disables), also checked
    before synthesis (429). Known limits: it counts successful battles only (a failed one
    still costs ~2 calls, uncounted but logged), and the non-transactional check may
    overshoot slightly under concurrency (no data is overwritten).
    """
    store = ArenaStore(pool)
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(422, "prompt must not be empty")

    pii = pii_match(prompt)
    if pii is not None:
        logger.info("arena_prompt_rejected", reason="pii", kind=pii)
        raise HTTPException(422, "prompt must not contain payment card or national id numbers")

    verdict = await moderation_verdict(settings, prompt)
    if verdict.flagged:
        logger.info(
            "arena_prompt_rejected",
            reason="moderation",
            categories=sorted(name for name, score in verdict.scores.items() if score >= 0.5),
        )
        raise HTTPException(422, "prompt rejected by content moderation")
    if not verdict.available and settings.arena_moderation_fail_closed:
        logger.warning("arena_prompt_rejected", reason="moderation_unavailable")
        raise HTTPException(503, "content moderation is unavailable, please retry")

    cap = settings.arena_daily_battle_cap
    if cap > 0 and await store.count_battles_today() >= cap:
        capture_api_event(
            posthog_client,
            "arena_battle_cap_reached",
            {"$process_person_profile": False},
        )
        raise HTTPException(429, "daily generation limit reached")

    # Whichever gender has fewer battles goes next, so the two accrue votes at
    # the same rate and neither board lags the other into significance.
    gender = gender_for_next_battle(await store.count_battles_by_gender())
    models = roster_for(gender)
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
        gender=gender,
    )
    if battle is None:
        raise HTTPException(502, "audio synthesis failed for one or both models")

    capture_api_event(
        posthog_client,
        "arena_battle_generated",
        {"$process_person_profile": False},
    )
    return await _battle_out(settings, battle.model_dump())


@router.get("/arena/battle/{battle_id}/reveal", response_model=RevealOut)
@limiter.limit("60/minute")
async def reveal_battle(
    request: Request,  # required by slowapi
    battle_id: uuid.UUID,
    voter_id: str = Query(...),
    x_labeler_key: str | None = Header(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    settings: Settings = Depends(get_settings),
) -> RevealOut:
    """De-anonymize a battle's two sides. Labeler-only, and only after this voter has voted."""
    if not _is_authenticated_labeler(x_labeler_key, settings):
        raise HTTPException(403, "reveal is not enabled")
    store = ArenaStore(pool)
    battle = await store.get_battle(battle_id)
    if battle is None:
        raise HTTPException(404, f"battle {battle_id} not found")
    if not await store.has_voted(battle_id=battle_id, voter_id=voter_id):
        raise HTTPException(409, "you have not voted on this battle yet")
    return RevealOut(
        a=RevealModelOut(provider=battle.provider_a, model=battle.model_a, label=battle.model_a),
        b=RevealModelOut(provider=battle.provider_b, model=battle.model_b, label=battle.model_b),
    )


@router.get("/arena/admin/cooccurrence", dependencies=[Depends(require_labeler)])
@limiter.limit("60/minute")
async def get_arena_cooccurrence(
    request: Request,  # required by slowapi
    metric: str = Query(default=PAIRING_METRIC),
    domain: str = Query(default=PAIRING_DOMAIN),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
) -> HTMLResponse:
    """The who-battled-whom heatmap. Reveals model identities, hence labeler-only."""
    store = ArenaStore(pool)
    rows = await store.get_cooccurrence_counts()
    ratings = await store.get_latest_ratings(metric_name=metric, domain=domain)
    cooc = build_cooccurrence(rows, {key: r.rating_elo for key, r in ratings.items()})
    capture_api_event(
        posthog_client,
        "arena_admin_report_viewed",
        {"report": "cooccurrence", "$process_person_profile": False},
    )
    return HTMLResponse(render_cooccurrence(cooc))


@router.get("/arena/admin/convergence", dependencies=[Depends(require_labeler)])
@limiter.limit("60/minute")
async def get_arena_convergence(
    request: Request,  # required by slowapi
    metric: str = Query(default=PAIRING_METRIC),
    domain: str = Query(default=PAIRING_DOMAIN),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
) -> HTMLResponse:
    """Per-model CI-vs-votes curves from snapshot history. Labeler-only."""
    rows = await ArenaStore(pool).get_snapshot_history(metric_name=metric, domain=domain)
    conv = build_convergence(rows)
    capture_api_event(
        posthog_client,
        "arena_admin_report_viewed",
        {"report": "convergence", "$process_person_profile": False},
    )
    return HTMLResponse(render_convergence(conv))
