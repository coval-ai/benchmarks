# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Who may see what on ``/v3``.

The proof is a bearer Clerk session token. The coval org sees everything. An org set
on a company sees that company's unpublished models, and partner-level metrics on that
company's models. Anyone else gets the public view. Which org belongs to which company
is read from the ``companies`` table on each request, so a change takes effect at once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import psycopg
from fastapi import Depends, Header, HTTPException, Response
from psycopg_pool import AsyncConnectionPool

from coval_bench.api import clerk
from coval_bench.api.deps import get_metric_visibility, get_models, get_pool, get_settings
from coval_bench.api.internal import EA_STATUS_HEADER, PUBLIC_CACHE_CONTROL, VARY_HEADERS
from coval_bench.config import Settings
from coval_bench.registries import RegisteredModel


@dataclass(frozen=True)
class Caller:
    """Who is asking: their level, and the company whose unpublished models they see."""

    proof: Literal["absent", "unknown", "accepted"]
    level: clerk.CallerLevel = "public"
    company: str | None = None


Visibility = Literal["public", "partner", "coval"]

_LOOSEST_FIRST: tuple[Visibility, ...] = ("public", "partner", "coval")


def strictest(*levels: Visibility) -> Visibility:
    """The level a result takes from its parts: the model, the metric and so on."""
    return max(levels, key=_LOOSEST_FIRST.index)


def can_see(who: Caller, level: Visibility, owner: str) -> bool:
    """May this caller see a result at *level* for a model that *owner* owns.

    Public is for everyone and coval is for coval staff. Partner is for coval staff
    and the owning company, so a partner never sees it on someone else's model.
    """
    if level == "public" or who.level == "coval":
        return True
    return level == "partner" and who.company == owner


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


def visible(
    response: Response,
    authorization: str | None = Header(default=None),
    who: Caller = Depends(caller),
    models: Sequence[RegisteredModel] = Depends(get_models),
    metric_levels: dict[str, str] = Depends(get_metric_visibility),
) -> Callable[[str, str, str], bool]:
    """A check of whether this caller may see a model's result for a metric.

    A result takes the stricter level of its model and its metric. A metric with no
    row in the metrics table is served to no one. Sets the cache headers too, so no
    route can serve a hidden result without them.
    """
    response.headers.append("Vary", VARY_HEADERS)
    response.headers[EA_STATUS_HEADER] = who.proof
    if who.proof == "accepted":
        response.headers["Cache-Control"] = "private, no-store"
    elif authorization is None:
        response.headers["Cache-Control"] = PUBLIC_CACHE_CONTROL
    unpublished = {(m.provider, m.model) for m in models if not m.published}

    def sees(provider: str, model: str, metric: str) -> bool:
        if metric not in metric_levels:
            return False
        model_level: Visibility = "partner" if (provider, model) in unpublished else "public"
        level = strictest(model_level, cast("Visibility", metric_levels[metric]))
        return can_see(who, level, provider)

    return sees
