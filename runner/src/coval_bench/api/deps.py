# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependency functions for the Coval Benchmarks API.

Routers import these helpers via ``Depends(get_pool)`` and
``Depends(get_settings)``. Both read from ``request.app.state``, which is
populated during the FastAPI lifespan (see ``app.py``).
"""

from __future__ import annotations

from typing import Any, cast

import structlog
from fastapi import HTTPException
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.config import Settings

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


def get_posthog(request: Request) -> Posthog | None:
    """Return the PostHog client from app state, or None if analytics is disabled."""
    return cast("Posthog | None", request.app.state.posthog)


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
