# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Who may see what on ``/v3``.

The proof is a bearer token: a Clerk session token, or a Clerk API key that an org
owns. The coval org sees everything. An org set on a company sees that company's
partner-level models, and partner-level metrics on that company's models. Anyone else
gets the public view. Which org belongs to which company is read from the ``companies``
table on each request, so a change takes effect at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import psycopg
from fastapi import Depends, Header, HTTPException, Response
from psycopg_pool import AsyncConnectionPool

from coval_bench.api import clerk
from coval_bench.api.deps import (
    bearer_token,
    get_metric_visibility,
    get_models_by_id,
    get_pool,
    get_settings,
)
from coval_bench.api.internal import EA_STATUS_HEADER, PUBLIC_CACHE_CONTROL, VARY_HEADERS
from coval_bench.config import Settings
from coval_bench.registries import RegisteredModel, Visibility


@dataclass(frozen=True)
class Caller:
    """Who is asking: their level, and the company whose partner-level results they see."""

    proof: Literal["absent", "unknown", "accepted"]
    level: clerk.CallerLevel = "public"
    company: str | None = None
    org_id: str | None = None


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
    """Resolve the caller. A signed-in request gets a 503 if its key or company can't be
    checked: the public view would hide a partner's own models without an error."""
    if authorization is None or settings.clerk_issuer is None:
        return Caller("absent")
    token = bearer_token(authorization)
    org_id: object
    if token is not None and token.count(".") != 2:
        if settings.clerk_secret_key is None:
            return Caller("unknown")
        try:
            owner = await clerk.api_key_owner(token, settings.clerk_secret_key)
        except clerk.UnknownApiKey:
            return Caller("unknown")
        except Exception as exc:
            raise HTTPException(503, "access is unavailable") from exc
        org_id = owner if owner.startswith("org_") else None
    else:
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
        return Caller("accepted", "partner", row["slug"], org_id)
    if settings.clerk_coval_org and org_id == settings.clerk_coval_org:
        return Caller("accepted", "coval", org_id=org_id)
    return Caller("accepted", org_id=org_id)


def coval_only(response: Response, who: Caller = Depends(caller)) -> Caller:
    """Let only coval staff through, and keep the response out of shared caches."""
    if who.level != "coval":
        raise HTTPException(403, "this endpoint is for coval staff")
    response.headers.append("Vary", VARY_HEADERS)
    response.headers["Cache-Control"] = "private, no-store"
    return who


def writer(who: Caller) -> str:
    """The org a write is recorded against."""
    if who.org_id is None:  # pragma: no cover — a coval caller always has its org
        raise HTTPException(403, "this endpoint is for coval staff")
    return who.org_id


def viewer(
    response: Response,
    authorization: str | None = Header(default=None),
    who: Caller = Depends(caller),
) -> Caller:
    """The caller of a read that is filtered by who is asking.

    Sets the cache headers, so no route can serve a hidden row without them.
    """
    response.headers.append("Vary", VARY_HEADERS)
    response.headers[EA_STATUS_HEADER] = who.proof
    if who.proof == "accepted":
        response.headers["Cache-Control"] = "private, no-store"
    elif authorization is None:
        response.headers["Cache-Control"] = PUBLIC_CACHE_CONTROL
    return who


class Sees(Protocol):
    """``visible``'s check. *benchmark* is the level of the result's benchmark, if it has one."""

    def __call__(self, model_id: int, metric: str, benchmark: Visibility = ...) -> bool: ...


def visible(
    who: Caller = Depends(viewer),
    models: dict[int, RegisteredModel] = Depends(get_models_by_id),
    metric_levels: dict[str, str] = Depends(get_metric_visibility),
) -> Sees:
    """A check of whether this caller may see a model's result for a metric.

    A result takes the strictest level of its model, its metric and, when it has one,
    its benchmark. A metric with no row in the metrics table is served to no one.
    """

    def sees(model_id: int, metric: str, benchmark: Visibility = "public") -> bool:
        model = models.get(model_id)
        if model is None or metric not in metric_levels:
            return False
        level = strictest(model.visibility, cast("Visibility", metric_levels[metric]), benchmark)
        return can_see(who, level, model.provider)

    return sees
