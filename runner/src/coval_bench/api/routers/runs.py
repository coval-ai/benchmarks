# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/runs — paginated list of benchmark runs.

Cursor-based pagination using ``before=<id>`` (constant-time index seek vs
slow offset pagination as data grows).
"""

from __future__ import annotations

from typing import Any

import psycopg.rows
import structlog
from fastapi import APIRouter, Depends, Query
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import RunOut

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["runs"])

_SQL = """
SELECT id, started_at, finished_at, upper(status) AS status,
       runner_sha, dataset_id, dataset_sha256, error
FROM benchmarks_v2.runs
WHERE %(before)s::bigint IS NULL OR id < %(before)s
ORDER BY id DESC
LIMIT %(limit)s
"""


@router.get("/runs")
@limiter.limit("60/minute")
async def list_runs(
    request: Request,  # required by slowapi for rate-limiting
    limit: int = Query(default=50, ge=1, le=200),
    before: int | None = Query(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> dict[str, Any]:
    """Return a newest-first page of benchmark runs.

    Args:
        limit: Maximum number of runs to return (1–200, default 50).
        before: Cursor — ``id`` of the last run on the previous page.

    Returns:
        ``{"runs": [...], "next_before": int | None}`` where ``next_before``
        is the smallest ``id`` in this page when there are exactly ``limit``
        rows, else ``None``.
    """
    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = await conn.execute(_SQL, {"before": before, "limit": limit})
        run_rows = await rows.fetchall()

    runs = [RunOut.model_validate(r) for r in run_rows]
    next_before: int | None = None
    if len(runs) == limit:
        next_before = min(r.id for r in runs)
    return {"runs": [r.model_dump(mode="json") for r in runs], "next_before": next_before}
