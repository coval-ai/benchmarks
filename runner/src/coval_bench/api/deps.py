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
from fastapi import HTTPException
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.cache import get_or_fill
from coval_bench.config import Settings
from coval_bench.db.model_state import ModelKey, ModelState, fetch_model_states

logger = structlog.get_logger("coval_bench.api")

# Cache key for the one model-state map entry (namespaced tuple, like the
# response cache keys).
MODEL_STATE_CACHE_KEY = ("model_states",)


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


def get_posthog(request: Request) -> Posthog | None:
    """Return the PostHog client from app state, or None if analytics is disabled."""
    return cast("Posthog | None", request.app.state.posthog)


def get_cache(request: Request) -> TTLCache[Any, Any]:
    """Return the per-app response TTL cache from app state."""
    return cast("TTLCache[Any, Any]", request.app.state.response_cache)


def get_cache_locks(request: Request) -> defaultdict[Any, asyncio.Lock]:
    """Return the per-app cache-key locks from app state."""
    return cast("defaultdict[Any, asyncio.Lock]", request.app.state.cache_locks)


async def get_model_states(request: Request) -> dict[ModelKey, ModelState]:
    """The state of every registry model, cached briefly per instance.

    The short TTL (``cache.MODEL_STATE_TTL_SECONDS``) is the ceiling on how
    long an admin toggle takes to reach every endpoint of this instance.
    """
    pool = await get_pool(request)
    cache = cast("TTLCache[Any, Any]", request.app.state.model_state_cache)
    locks = get_cache_locks(request)
    states, _ = await get_or_fill(
        cache, locks, MODEL_STATE_CACHE_KEY, lambda: fetch_model_states(pool)
    )
    return states


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
