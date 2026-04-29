# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Health check endpoints.

``GET /healthz`` — liveness probe. No DB hit. Always answers.
``GET /readyz`` — readiness probe. Acquires a DB connection and runs SELECT 1.

Both endpoints are **exempt from rate limiting** — they are polled by the
GCP load balancer / Cloud Run health check and must never return 429.
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
    try:
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
        return JSONResponse({"status": "ready"})
    except Exception as exc:  # noqa: BLE001
        logger.warning("readyz_db_unreachable", error=str(exc)[:200])
        return JSONResponse(
            {"status": "not ready", "error": str(exc)[:200]},
            status_code=503,
        )
