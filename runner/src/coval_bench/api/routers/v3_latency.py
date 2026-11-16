# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/latency: each model's latency over a time range, from the latency probe."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.common import WINDOW_DURATIONS, WindowLiteral
from coval_bench.api.deps import get_pool
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ErrorResponse, LatencyOut, LatencyResponse
from coval_bench.db.probe_store import fetch_latency
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
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
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
            (row["provider"], row["model"]) not in hidden
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
