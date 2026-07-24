# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Health check endpoints.

``GET /healthz`` — liveness probe. No DB hit. Always answers.
``GET /readyz`` — readiness probe. Acquires a DB connection and runs SELECT 1.
``GET /v1/health`` — public health check for API consumers. Same DB probe as
``/readyz``, without the internal error string in the body.

All three are **exempt from rate limiting** — they are polled by the GCP load
balancer / Cloud Run health check and by external uptime monitors, and must
never return 429.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from psycopg_pool import AsyncConnectionPool

from coval_bench.api.deps import get_pool

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["health"])
v1_router = APIRouter(tags=["health"])


async def _probe_db(pool: AsyncConnectionPool[Any]) -> str | None:
    """Run SELECT 1. Returns ``None`` when reachable, else a truncated error."""
    try:
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        logger.warning("health_db_unreachable", error=str(exc)[:200])
        return str(exc)[:200]
    return None


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
    error = await _probe_db(pool)
    if error is None:
        return JSONResponse({"status": "ready"})
    return JSONResponse({"status": "not ready", "error": error}, status_code=503)


@v1_router.get("/health")
async def health(
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> JSONResponse:
    """Public health check — 200 when the API can serve data, 503 otherwise.

    The error string stays out of the body: this path is publicly reachable,
    while ``/readyz`` exists for operators.
    """
    error = await _probe_db(pool)
    if error is None:
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "unavailable"}, status_code=503)
