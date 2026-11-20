# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""/v3/metrics: who sees each metric and whether it is collected. Coval staff write."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

import psycopg
import structlog
from fastapi import APIRouter, Depends, HTTPException
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool
from coval_bench.api.internal import _VISIBLE_TO
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ErrorResponse, MetricOut, MetricPatch, MetricsResponse
from coval_bench.api.v3_access import Caller, coval_only, viewer, writer
from coval_bench.db.metric_store import MetricStore, RegisteredMetric

router = APIRouter(tags=["metrics"])
logger = structlog.get_logger("coval_bench.api.v3_metrics")

_WRITE_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (400, 403, 404)
}


async def _guarded[T](call: Awaitable[T]) -> T:
    try:
        return await call
    except psycopg.Error as exc:
        raise HTTPException(503, "metrics are unavailable") from exc


def _out(metric: RegisteredMetric, who: Caller) -> MetricOut:
    return MetricOut(
        code=metric.code,
        display_name=metric.display_name,
        visibility=metric.visibility,
        collected=metric.collected if who.level == "coval" else None,
    )


@router.get("/metrics", response_model=MetricsResponse)
@limiter.limit("60/minute")
async def get_metrics(
    request: Request,  # required by slowapi
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    who: Caller = Depends(viewer),
) -> MetricsResponse:
    """Every metric the caller may see, in code order.

    The public sees public metrics, a partner org also sees partner-level ones, and
    coval staff see every metric with whether it is collected.
    """
    metrics = await _guarded(MetricStore(pool).metrics())
    return MetricsResponse(
        metrics=[_out(m, who) for m in metrics if m.visibility in _VISIBLE_TO[who.level]]
    )


@router.patch("/metrics/{code}", response_model=MetricOut, responses=_WRITE_ERRORS)
@limiter.limit("60/minute")
async def update(
    request: Request,  # required by slowapi
    code: str,
    body: MetricPatch,
    who: Caller = Depends(coval_only),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> MetricOut:
    """Change who sees a metric, whether it is collected, or both. Coval staff only."""
    org = writer(who)
    if body.visibility is None and body.collected is None:
        raise HTTPException(400, "give visibility, collected or both")
    metric = await _guarded(
        MetricStore(pool).update(code, visibility=body.visibility, collected=body.collected)
    )
    if metric is None:
        raise HTTPException(404, "no such metric")
    logger.info(
        "metric_changed",
        code=metric.code,
        visibility=metric.visibility,
        collected=metric.collected,
        org=org,
    )
    return _out(metric, who)
