# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/{modality}/latency: each model's latency over a time range, from the latency probe."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import psycopg
from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api import v3
from coval_bench.api.cache import get_or_fill
from coval_bench.api.common import WINDOW_DURATIONS, StatisticLiteral, WindowLiteral
from coval_bench.api.deps import (
    get_cache,
    get_cache_locks,
    get_models_by_id,
    get_pool,
    get_settings,
)
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    ErrorResponse,
    LatencyOut,
    LatencyPointOut,
    LatencyPointsResponse,
    LatencyResponse,
    ProbeOut,
    ProbesResponse,
    SweepLatencyOut,
    SweepLatencyResponse,
)
from coval_bench.api.v3_access import Sees, coval_only, visible
from coval_bench.config import Settings
from coval_bench.db.probe_store import fetch_latency, fetch_latency_points, fetch_probes
from coval_bench.db.sweep_store import fetch_sweep_latency
from coval_bench.registries import Benchmark, RegisteredModel, is_metric_excluded

router = APIRouter(tags=["latency"])

MetricLiteral = Literal["TTFT", "TTFS", "TTFA"]

_METRICS: dict[str, tuple[MetricLiteral, ...]] = {"STT": ("TTFT", "TTFS"), "TTS": ("TTFA",)}
_FRACTIONS: dict[str, float | None] = {
    "default": None,
    "p50": 0.5,
    "p90": 0.9,
    "p95": 0.95,
    "p100": 1.0,
}


@router.get("/latency", response_model=LatencyResponse, responses={400: {"model": ErrorResponse}})
@limiter.limit("60/minute")
async def get_latency(
    request: Request,  # required by slowapi
    modality: Benchmark = Depends(v3.path_modality),
    metric: MetricLiteral | None = Query(default=None),
    window: WindowLiteral = Query(default="24h"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    sees: Sees = Depends(visible),
) -> LatencyResponse:
    """One row per model and metric: percentiles over the probes that succeeded in the window.

    ``sample_count`` and ``failed_count`` together are the model's availability. Only
    failures on the provider's side count. A probe that failed because of our key, our
    credit or our rate limit is in neither. ``failed_count`` is the model's, so it is the
    same on each of a model's rows and adding it up across them counts it twice.
    """
    try:
        stats, failures = await fetch_latency(
            pool, since=datetime.now(UTC) - WINDOW_DURATIONS[window]
        )
    except psycopg.Error as exc:
        raise HTTPException(503, "latency results are unavailable") from exc

    failed = {row["model_id"]: row["failed_count"] for row in failures}

    def wanted(row: dict[str, Any], code: str) -> bool:
        return (
            sees(row["model_id"], code)
            and row["modality"] == modality
            and metric in (None, code)
            and not is_metric_excluded(row["provider"], row["model"], code)
        )

    rows = [
        LatencyOut(
            model_id=row["model_id"],
            provider=row["provider"],
            model=row["model"],
            metric=row["metric"],
            p25=round(row["p25"], 1),
            p50=round(row["p50"], 1),
            p75=round(row["p75"], 1),
            p90=round(row["p90"], 1),
            p95=round(row["p95"], 1),
            mean=round(row["mean"], 1),
            min=round(row["min"], 1),
            max=round(row["max"], 1),
            sample_count=row["sample_count"],
            failed_count=failed.get(row["model_id"], 0),
        )
        for row in stats
        if wanted(row, row["metric"])
    ]

    measured = {row["model_id"] for row in stats}
    for row in failures:
        if row["model_id"] in measured:
            continue
        for code in _METRICS.get(row["modality"], ()):
            if wanted(row, code):
                rows.append(
                    LatencyOut(
                        model_id=row["model_id"],
                        provider=row["provider"],
                        model=row["model"],
                        metric=code,
                        sample_count=0,
                        failed_count=row["failed_count"],
                    )
                )

    rows.sort(key=lambda row: (row.model_id, row.metric))
    return LatencyResponse(latency=rows)


@router.get(
    "/latency/sweep",
    response_model=SweepLatencyResponse,
    responses={400: {"model": ErrorResponse}},
)
@limiter.limit("60/minute")
async def get_sweep_latency(
    request: Request,  # required by slowapi
    modality: Benchmark = Depends(v3.path_modality),
    metric: MetricLiteral | None = Query(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    sees: Sees = Depends(visible),
    cache: TTLCache[Any, Any] = Depends(get_cache),
    cache_locks: defaultdict[Any, asyncio.Lock] = Depends(get_cache_locks),
) -> SweepLatencyResponse:
    """One row per model and metric: percentiles over the items of its quality sweep.

    This is the latency the model had while its quality scores were measured. It covers
    the model's completed sweeps at its current version, so it has no time range and
    changes only when the model is swept again. The rows are cached for every caller;
    what each caller may see is decided after.
    """
    try:
        stats, _ = await get_or_fill(
            cache, cache_locks, ("v3-sweep-latency",), lambda: fetch_sweep_latency(pool)
        )
    except psycopg.Error as exc:
        raise HTTPException(503, "latency results are unavailable") from exc

    rows = [
        SweepLatencyOut(
            model_id=row["model_id"],
            provider=row["provider"],
            model=row["model"],
            metric=row["metric"],
            p25=round(row["p25"], 1),
            p50=round(row["p50"], 1),
            p75=round(row["p75"], 1),
            p90=round(row["p90"], 1),
            p95=round(row["p95"], 1),
            mean=round(row["mean"], 1),
            min=round(row["min"], 1),
            max=round(row["max"], 1),
            sample_count=row["sample_count"],
        )
        for row in stats
        if sees(row["model_id"], row["metric"])
        and row["modality"] == modality
        and metric in (None, row["metric"])
        and not is_metric_excluded(row["provider"], row["model"], row["metric"])
    ]
    rows.sort(key=lambda row: (row.model_id, row.metric))
    return SweepLatencyResponse(latency=rows)


@router.get(
    "/latency/points",
    response_model=LatencyPointsResponse,
    responses={400: {"model": ErrorResponse}},
)
@limiter.limit("60/minute")
async def get_latency_points(
    request: Request,  # required by slowapi
    modality: Benchmark = Depends(v3.path_modality),
    metric: MetricLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    statistic: StatisticLiteral = Query(default="default"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    settings: Settings = Depends(get_settings),
    sees: Sees = Depends(visible),
) -> LatencyPointsResponse:
    """Every model of the modality's *metric* over the window, as points for a chart.

    The last 24 hours has one point per probe tick. Seven days is bucketed by the hour and
    thirty by the day, and ``bucket_seconds`` says which. A point is the mean of its
    bucket's probes, or with ``statistic`` a percentile or (``p100``) the maximum; a tick
    is one probe, so every statistic gives the same 24-hour points. A point with a null value is a
    tick or bucket where every probe failed on the provider's side. A metric the modality
    doesn't have has no points.
    """
    bucket = {
        "24h": timedelta(seconds=settings.schedule_period_seconds),
        "7d": timedelta(hours=1),
        "30d": timedelta(days=1),
    }[window]
    if metric not in _METRICS.get(modality, ()):
        return LatencyPointsResponse(
            metric=metric,
            statistic=statistic,
            bucket_seconds=int(bucket.total_seconds()),
            points=[],
        )
    try:
        rows = await fetch_latency_points(
            pool,
            modality=modality,
            metric=metric,
            since=datetime.now(UTC) - WINDOW_DURATIONS[window],
            bucket=bucket,
            fraction=_FRACTIONS[statistic],
        )
    except psycopg.Error as exc:
        raise HTTPException(503, "latency results are unavailable") from exc

    points = [
        LatencyPointOut(
            model_id=row["model_id"],
            provider=row["provider"],
            model=row["model"],
            scheduled_at=row["scheduled_at"].astimezone(UTC),
            value=None if row["value"] is None else round(row["value"], 1),
        )
        for row in rows
        if sees(row["model_id"], metric)
        and not is_metric_excluded(row["provider"], row["model"], metric)
    ]
    return LatencyPointsResponse(
        metric=metric,
        statistic=statistic,
        bucket_seconds=int(bucket.total_seconds()),
        points=points,
    )


@router.get(
    "/latency/probes",
    response_model=ProbesResponse,
    responses={400: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
    dependencies=[Depends(coval_only)],
)
@limiter.limit("60/minute")
async def get_probes(
    request: Request,  # required by slowapi
    modality: Benchmark = Depends(v3.path_modality),
    model_id: int = Query(...),
    window: WindowLiteral = Query(default="24h"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    models: dict[int, RegisteredModel] = Depends(get_models_by_id),
) -> ProbesResponse:
    """Every probe of one model over the window, oldest first. Coval staff only.

    Unlike ``/latency``, a probe that failed on our side is listed too, with a
    ``failure_origin`` of ``runner``.
    """
    model = models.get(model_id)
    if model is None or model.benchmark != modality:
        raise HTTPException(404, "no such model in this modality")
    try:
        rows = await fetch_probes(
            pool, model_id=model_id, since=datetime.now(UTC) - WINDOW_DURATIONS[window]
        )
    except psycopg.Error as exc:
        raise HTTPException(503, "latency results are unavailable") from exc
    return ProbesResponse(probes=[ProbeOut(**row) for row in rows])
