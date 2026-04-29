# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/results — filtered list of benchmark results.

Always filters ``status = 'success'``. Supports optional filters for provider,
model, metric type, benchmark type, and time window.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import psycopg.rows
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ResultOut

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["results"])

MetricLiteral = Literal["WER", "TTFA", "TTFT", "RTF", "AUDIO_TO_FINAL"]
BenchmarkLiteral = Literal["STT", "TTS"]


@router.get("/results")
@limiter.limit("60/minute")
async def list_results(
    request: Request,  # required by slowapi
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    metric: MetricLiteral | None = Query(default=None),
    benchmark: BenchmarkLiteral | None = Query(default=None),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> dict[str, Any]:
    """Return a newest-first page of successful benchmark results.

    All optional filters are ANDed together. Rejected if ``since > until``.
    Only rows with ``status = 'success'`` are returned.
    """
    if since is not None and until is not None and since > until:
        raise HTTPException(400, "since must not be after until")

    # Build WHERE clause dynamically — parameterised only, no f-string SQL injection.
    conditions: list[str] = ["r.status = 'success'"]
    params: dict[str, Any] = {"limit": limit}

    if provider is not None:
        conditions.append("r.provider = %(provider)s")
        params["provider"] = provider
    if model is not None:
        conditions.append("r.model = %(model)s")
        params["model"] = model
    if metric is not None:
        conditions.append("r.metric_type = %(metric)s")
        params["metric"] = metric
    if benchmark is not None:
        conditions.append("r.benchmark = %(benchmark)s")
        params["benchmark"] = benchmark
    if since is not None:
        conditions.append("r.created_at >= %(since)s")
        params["since"] = since
    if until is not None:
        conditions.append("r.created_at <= %(until)s")
        params["until"] = until

    where_clause = " AND ".join(conditions)
    # S608 false-positive: where_clause contains only pre-defined SQL fragments
    # (constants from the ``conditions`` list above), never raw user input.
    # The f-string interpolates only the WHERE clause built from static strings.
    select = (
        "SELECT r.id, r.run_id, r.provider, r.model, r.voice, r.benchmark,"
        " r.metric_type, r.metric_value, r.metric_units, r.audio_filename,"
        " r.created_at FROM benchmarks_v2.results r"
    )
    sql = f"{select} WHERE {where_clause} ORDER BY r.created_at DESC, r.id DESC LIMIT %(limit)s"  # noqa: S608

    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = await conn.execute(sql, params)
        result_rows = await rows.fetchall()

    results = [ResultOut.model_validate(r) for r in result_rows]
    return {"results": [r.model_dump(mode="json") for r in results]}
