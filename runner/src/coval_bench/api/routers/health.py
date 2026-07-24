# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Health check endpoints.

``GET /healthz`` — liveness probe. No DB hit. Always answers.
``GET /readyz`` — readiness probe. Acquires a DB connection and runs SELECT 1.
``GET /v1/health`` — public health check for API consumers. Same DB probe as
``/readyz``, rate-limited like the rest of ``/v1``.

``/healthz`` and ``/readyz`` are **exempt from rate limiting** — they are polled
by the GCP load balancer / Cloud Run health check and must never return 429.
``/v1/health`` carries the standard 60/minute limit: it is publicly advertised
and hits the 4-connection pool shared with the data routes, and 60/minute leaves
ample headroom for an uptime monitor polling every few seconds.

Neither DB-backed route returns the underlying error to the caller — both paths
are publicly reachable and psycopg errors can carry connection details. The
error is logged instead (``health_db_unreachable``).
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool
from coval_bench.api.ratelimit import limiter

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["health"])
v1_router = APIRouter(tags=["health"])


async def _db_reachable(pool: AsyncConnectionPool[Any]) -> bool:
    """Run SELECT 1. Logs and returns ``False`` when the DB is unreachable."""
    try:
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
    except Exception:
        logger.warning("health_db_unreachable", exc_info=True)
        return False
    return True


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe — always returns 200."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> JSONResponse:
    """Readiness probe — returns 200 if DB is reachable, 503 otherwise.

    Never raises an exception — DB unreachable is the expected failure mode
    during Cloud Run startup.
    """
    if await _db_reachable(pool):
        return JSONResponse({"status": "ready"})
    return JSONResponse({"status": "not ready"}, status_code=503)


@v1_router.get("/health")
@limiter.limit("60/minute")
async def health(
    request: Request,
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> JSONResponse:
    """Public health check — 200 when the API can serve data, 503 otherwise."""
    if await _db_reachable(pool):
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "unavailable"}, status_code=503)
