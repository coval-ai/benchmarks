# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependency functions for the Coval Benchmarks API.

Routers import these helpers via ``Depends(get_pool)`` and
``Depends(get_settings)``. Both read from ``request.app.state``, which is
populated during the FastAPI lifespan (see ``app.py``).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, cast

import structlog
from cachetools import TTLCache
from fastapi import Depends, Header, HTTPException
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api import clerk
from coval_bench.config import Settings
from coval_bench.db.registry_store import TagRecord, fetch_models, fetch_tags
from coval_bench.registries import RegisteredModel

logger = structlog.get_logger("coval_bench.api")


async def get_pool(request: Request) -> AsyncConnectionPool[Any]:
    """Return the async connection pool from app state.

    Raises 503 if the pool was never initialised (e.g. during startup).
    """
    pool: AsyncConnectionPool[Any] | None = request.app.state.pool
    if pool is None:
        raise HTTPException(503, "database pool not initialised")
    return pool


def get_settings(request: Request) -> Settings:
    """Return the Settings instance from app state."""
    settings: Settings = request.app.state.settings
    return settings


def require_coval_admin(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> clerk.CovalAdmin:
    """The verified coval caller: 401 without a proven token, 403 outside the coval org."""
    claims = clerk.bearer_claims(authorization, settings)
    if claims is None:
        raise HTTPException(
            401,
            "a valid Clerk session token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(
            401,
            "the session token names no subject",
            headers={"WWW-Authenticate": "Bearer"},
        )
    org_id = claims.get("org_id")
    if (
        not isinstance(org_id, str)
        or not settings.clerk_coval_org
        or org_id != settings.clerk_coval_org
    ):
        raise HTTPException(403, "the coval org must be active on the token")
    email = claims.get("email")
    return clerk.CovalAdmin(
        user_id=user_id,
        email=email if isinstance(email, str) and email else None,
    )


def get_posthog(request: Request) -> Posthog | None:
    """Return the PostHog client from app state, or None if analytics is disabled."""
    return cast("Posthog | None", request.app.state.posthog)


def get_cache(request: Request) -> TTLCache[Any, Any]:
    """Return the per-app response TTL cache from app state."""
    return cast("TTLCache[Any, Any]", request.app.state.response_cache)


async def get_models(
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> list[RegisteredModel]:
    """The model roster, read fresh on every request.

    Deliberately uncached. The roster decides who may see an embargoed model,
    and a cached copy is per-instance: unpublishing one would stay visible on
    every other instance until its copy aged out. The read is a few
    milliseconds against a table of this size.

    Raises 503 rather than falling back to a default: a made-up roster could
    serve an embargoed model to the public.
    """
    try:
        return await fetch_models(pool)
    except Exception as exc:
        logger.error("model_roster_unavailable", exc_info=True)
        raise HTTPException(503, "the model registry is unavailable") from exc


async def get_tags(
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> dict[str, TagRecord]:
    """The tag vocabulary, read fresh so a newly added tag surfaces at once."""
    try:
        return await fetch_tags(pool)
    except Exception as exc:
        logger.error("tag_vocabulary_unavailable", exc_info=True)
        raise HTTPException(503, "the tag vocabulary is unavailable") from exc


def get_cache_locks(request: Request) -> defaultdict[Any, asyncio.Lock]:
    """Return the per-app cache-key locks from app state."""
    return cast("defaultdict[Any, asyncio.Lock]", request.app.state.cache_locks)


def capture_api_event(client: Posthog | None, event: str, properties: dict[str, Any]) -> None:
    """Best-effort PostHog capture for API routes; never fails the request."""
    if client is None:
        return
    try:
        client.capture(
            event, distinct_id="coval-bench-api", properties=properties, disable_geoip=True
        )
    except Exception:
        logger.warning("posthog_capture_failed", event_name=event, exc_info=True)
