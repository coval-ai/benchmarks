# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/latency: each model's latency over a time range, from the latency probe."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.common import WINDOW_DURATIONS, WindowLiteral
from coval_bench.api.deps import get_pool, get_settings
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    ErrorResponse,
    LatencyOut,
    LatencyPointOut,
    LatencyPointsResponse,
    LatencyResponse,
)
from coval_bench.api.v3_access import visible
from coval_bench.config import Settings
from coval_bench.db.probe_store import fetch_latency, fetch_latency_points
from coval_bench.registries import is_metric_excluded

router = APIRouter(tags=["latency"])

ModalityLiteral = Literal["STT", "TTS"]
MetricLiteral = Literal["TTFT", "TTFS", "TTFA"]

_METRICS: dict[str, tuple[MetricLiteral, ...]] = {"STT": ("TTFT", "TTFS"), "TTS": ("TTFA",)}


@router.get("/latency", response_model=LatencyResponse, responses={400: {"model": ErrorResponse}})
@limiter.limit("60/minute")
async def get_latency(
    request: Request,  # required by slowapi
    modality: ModalityLiteral | None = Query(default=None),
    metric: MetricLiteral | None = Query(default=None),
    window: WindowLiteral = Query(default="24h"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    sees: Callable[[str, str, str, str], bool] = Depends(visible),
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
            sees(row["modality"], row["provider"], row["model"], code)
            and modality in (None, row["modality"])
            and metric in (None, code)
            and not is_metric_excluded(row["provider"], row["model"], code)
        )

    rows = [
        LatencyOut(
            model_id=row["model_id"],
            provider=row["provider"],
            model=row["model"],
            metric=row["metric"],
            p50=round(row["p50"], 1),
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
    "/latency/points",
    response_model=LatencyPointsResponse,
    responses={400: {"model": ErrorResponse}},
)
@limiter.limit("60/minute")
async def get_latency_points(
    request: Request,  # required by slowapi
    metric: MetricLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    settings: Settings = Depends(get_settings),
    sees: Callable[[str, str, str, str], bool] = Depends(visible),
) -> LatencyPointsResponse:
    """Every model's *metric* over the window, as points for a chart.

    The last 24 hours has one point per probe tick. Seven days is averaged by the hour and
    thirty by the day, and ``bucket_seconds`` says which. A point with a null value is a
    tick or bucket where every probe failed on the provider's side.
    """
    bucket = {
        "24h": timedelta(seconds=settings.schedule_period_seconds),
        "7d": timedelta(hours=1),
        "30d": timedelta(days=1),
    }[window]
    modality = "TTS" if metric == "TTFA" else "STT"
    try:
        rows = await fetch_latency_points(
            pool,
            modality=modality,
            metric=metric,
            since=datetime.now(UTC) - WINDOW_DURATIONS[window],
            bucket=bucket,
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
        if sees(modality, row["provider"], row["model"], metric)
        and not is_metric_excluded(row["provider"], row["model"], metric)
    ]
    return LatencyPointsResponse(
        metric=metric, bucket_seconds=int(bucket.total_seconds()), points=points
    )
