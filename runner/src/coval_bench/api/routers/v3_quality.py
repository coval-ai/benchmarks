# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/quality: each model's quality scores, from the quality sweep."""

from __future__ import annotations

from datetime import UTC
from typing import Any, Literal

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ErrorResponse, QualityOut, QualityResponse
from coval_bench.db.sweep_store import fetch_quality
from coval_bench.registries import is_metric_excluded

router = APIRouter(tags=["quality"])

ModalityLiteral = Literal["STT", "TTS"]


@router.get("/quality", response_model=QualityResponse, responses={400: {"model": ErrorResponse}})
@limiter.limit("60/minute")
async def get_quality(
    request: Request,  # required by slowapi
    modality: ModalityLiteral | None = Query(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> QualityResponse:
    """One row per model, dataset and metric, from the model's latest completed sweep.

    ``value`` is pooled: ``errors`` over ``reference_units`` across every item, as a
    percent. ``sample_count`` against ``expected_sample_count`` is how much of the dataset
    was scored. A model with no completed sweep is not listed.
    """
    try:
        rows = await fetch_quality(pool)
    except psycopg.Error as exc:
        raise HTTPException(503, "quality results are unavailable") from exc

    quality = [
        QualityOut(
            model_id=row["model_id"],
            provider=row["provider"],
            model=row["model"],
            model_version=row["model_version"],
            dataset=row["dataset"],
            dataset_version=row["dataset_version"],
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
        if (row["provider"], row["model"]) not in hidden
        and modality in (None, row["modality"])
        and not is_metric_excluded(row["provider"], row["model"], row["metric"])
    ]
    return QualityResponse(quality=quality)
