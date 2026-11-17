# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/quality: each model's quality scores, from the quality sweep."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from typing import Any, Literal

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ErrorResponse, QualityOut, QualityResponse
from coval_bench.api.v3_access import visible
from coval_bench.db.sweep_store import fetch_quality
from coval_bench.registries import is_metric_excluded

router = APIRouter(tags=["quality"])

ModalityLiteral = Literal["STT", "TTS"]

_POOL_KEY = (
    "model_id",
    "model_version",
    "benchmark",
    "benchmark_version",
    "metric",
    "metric_version",
)


def _pooled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model, benchmark and metric, summed over the benchmark's conditions.

    A metric version that didn't score every condition is dropped: its sum would
    cover only part of the benchmark.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in _POOL_KEY), []).append(row)
    return [
        {
            **group[0],
            "condition": None,
            "errors": sum(row["errors"] for row in group),
            "reference_units": sum(row["reference_units"] for row in group),
            "sample_count": sum(row["sample_count"] for row in group),
            "expected_sample_count": sum(row["expected_sample_count"] for row in group),
            "measured_at": max(row["measured_at"] for row in group),
        }
        for group in groups.values()
        if len(group) == group[0]["conditions"]
    ]


@router.get("/quality", response_model=QualityResponse, responses={400: {"model": ErrorResponse}})
@limiter.limit("60/minute")
async def get_quality(
    request: Request,  # required by slowapi
    modality: ModalityLiteral | None = Query(default=None),
    benchmark: str | None = Query(default=None),
    metric: str | None = Query(default=None),
    model_id: int | None = Query(default=None),
    breakdown: Literal["condition"] | None = Query(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    sees: Callable[[str, str, str], bool] = Depends(visible),
) -> QualityResponse:
    """One row per model, benchmark and metric, pooled across the benchmark's conditions.

    ``breakdown=condition`` returns one row per condition instead, each naming its
    ``condition``; a response never holds both levels. ``value`` is pooled: ``errors``
    over ``reference_units`` across every item, as a percent. ``sample_count`` against
    ``expected_sample_count`` is how much of the benchmark was scored, and
    ``benchmark_version`` is the version of its datasets.

    A model's rows come from its newest version with a completed sweep of every
    condition, so it keeps its previous scores until the last condition finishes and is
    not listed for a benchmark it has never finished. A sweep scored under more than one
    metric version has a row for each.
    """
    try:
        rows = await fetch_quality(pool)
    except psycopg.Error as exc:
        raise HTTPException(503, "quality results are unavailable") from exc

    rows = [
        row
        for row in rows
        if sees(row["provider"], row["model"], row["metric"])
        and modality in (None, row["modality"])
        and benchmark in (None, row["benchmark"])
        and metric in (None, row["metric"])
        and model_id in (None, row["model_id"])
        and not is_metric_excluded(row["provider"], row["model"], row["metric"])
    ]
    if breakdown is None:
        rows = _pooled(rows)
    quality = [
        QualityOut(
            model_id=row["model_id"],
            provider=row["provider"],
            model=row["model"],
            model_version=row["model_version"],
            benchmark=row["benchmark"],
            benchmark_version=row["benchmark_version"],
            condition=row["condition"],
            metric=row["metric"],
            metric_version=row["metric_version"],
            value=(
                round(100 * row["errors"] / row["reference_units"], 2)
                if row["reference_units"]
                else None
            ),
            errors=row["errors"],
            reference_units=row["reference_units"],
            sample_count=row["sample_count"],
            expected_sample_count=row["expected_sample_count"],
            measured_at=row["measured_at"].astimezone(UTC),
        )
        for row in rows
    ]
    return QualityResponse(quality=quality)
