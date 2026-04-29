# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependency functions for the Coval Benchmarks API.

Routers import these helpers via ``Depends(get_pool)`` and
``Depends(get_settings)``. Both read from ``request.app.state``, which is
populated during the FastAPI lifespan (see ``app.py``).
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.config import Settings


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
