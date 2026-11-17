# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Who may see what on ``/v3``.

The proof is a bearer Clerk session token. The coval org sees everything. An org set
on a company sees that company's unpublished models and partner-level metrics. Anyone
else gets the public view. Which org belongs to which company is read from the
``companies`` table on each request, so a change takes effect at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import psycopg
from fastapi import Depends, Header, HTTPException, Response
from psycopg_pool import AsyncConnectionPool

from coval_bench.api import clerk
from coval_bench.api.deps import get_metric_visibility, get_models, get_pool, get_settings
from coval_bench.api.internal import (
    _VISIBLE_TO,
    EA_STATUS_HEADER,
    PUBLIC_CACHE_CONTROL,
    VARY_HEADERS,
    embargoed_pairs,
    with_retired_keys,
)
from coval_bench.config import Settings
from coval_bench.registries import RegisteredModel


@dataclass(frozen=True)
class Caller:
    """Who is asking: their level, and the company whose unpublished models they see."""

    proof: Literal["absent", "unknown", "accepted"]
    level: clerk.CallerLevel = "public"
    company: str | None = None


async def caller(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> Caller:
    """Resolve the caller. A signed-in request gets a 503 if its company can't be read:
    answering with the public view would hide a partner's own models without an error."""
    if authorization is None or settings.clerk_issuer is None:
        return Caller("absent")
    claims = clerk.bearer_claims(authorization, settings)
    if claims is None:
        return Caller("unknown")
    org_id = claims.get("org_id")
    if not isinstance(org_id, str):
        return Caller("accepted")
    try:
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT slug FROM benchmarks_v2.companies WHERE clerk_org_id = %s", (org_id,)
                )
            ).fetchone()
    except psycopg.Error as exc:
        raise HTTPException(503, "access is unavailable") from exc
    if row is not None:
        return Caller("accepted", "partner", row["slug"])
    if settings.clerk_coval_org and org_id == settings.clerk_coval_org:
        return Caller("accepted", "coval")
    return Caller("accepted")


def hidden_models(
    response: Response,
    authorization: str | None = Header(default=None),
    who: Caller = Depends(caller),
    models: Sequence[RegisteredModel] = Depends(get_models),
) -> frozenset[tuple[str, str]]:
    """The ``(provider, model)`` pairs this caller's responses must not contain.

    Sets the cache headers too, so no route can serve embargoed rows without them.
    """
    response.headers.append("Vary", VARY_HEADERS)
    response.headers[EA_STATUS_HEADER] = who.proof
    embargoed = embargoed_pairs(models)
    if who.proof != "accepted":
        if authorization is None:
            response.headers["Cache-Control"] = PUBLIC_CACHE_CONTROL
        return embargoed
    response.headers["Cache-Control"] = "private, no-store"
    if who.level == "coval":
        return frozenset()
    own = frozenset(pair for pair in embargoed if pair[0] == who.company)
    return embargoed - with_retired_keys(own)


def visible_metrics(
    who: Caller = Depends(caller),
    visibility: dict[str, str] = Depends(get_metric_visibility),
    _cache_headers: frozenset[tuple[str, str]] = Depends(hidden_models),
) -> frozenset[str]:
    """The only metric codes this caller's responses may contain."""
    return frozenset(code for code, shown in visibility.items() if shown in _VISIBLE_TO[who.level])
