# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/sweeps: each quality sweep's progress and its items. Coval staff only."""

from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api import v3
from coval_bench.api.deps import get_pool
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    ErrorResponse,
    SweepItemOut,
    SweepItemsResponse,
    SweepOut,
    SweepsResponse,
)
from coval_bench.api.v3_access import coval_only
from coval_bench.db.sweep_store import fetch_sweep_items, fetch_sweeps
from coval_bench.registries import Benchmark

router = APIRouter(tags=["sweeps"], dependencies=[Depends(coval_only)])


@router.get("/sweeps", response_model=SweepsResponse, responses={403: {"model": ErrorResponse}})
@limiter.limit("60/minute")
async def get_sweeps(
    request: Request,  # required by slowapi
    modality: Benchmark = Depends(v3.path_modality),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> SweepsResponse:
    """One row per sweep of the modality, oldest first, finished or not.

    ``result_count`` items have run, ``queued_count`` are still to run and
    ``gave_up_count`` ran out of tries. A sweep with ``completed_at`` set has nothing
    left to run. A model has a sweep per dataset for each of its versions, so an old
    version's sweeps are listed too.
    """
    try:
        rows = await fetch_sweeps(pool)
    except psycopg.Error as exc:
        raise HTTPException(503, "sweeps are unavailable") from exc
    return SweepsResponse(sweeps=[SweepOut(**row) for row in rows if row["modality"] == modality])


@router.get(
    "/sweeps/{sweep_id}/items",
    response_model=SweepItemsResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
@limiter.limit("60/minute")
async def get_sweep_items(
    request: Request,  # required by slowapi
    sweep_id: int,
    modality: Benchmark = Depends(v3.path_modality),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> SweepItemsResponse:
    """Every item of a sweep in sample and voice order, the ones that haven't run included.

    A ``done`` item has the transcript, the timings, each metric's ``scores`` and, for
    TTS, the ``audio_uri`` of the audio the model made. For STT the transcript is the
    model's. For TTS it is what the model's audio was heard to say. A ``queued`` item
    hasn't run yet or is waiting to retry, and a ``gave_up`` item ran out of tries.
    """
    try:
        sweeps = await fetch_sweeps(pool)
        mine = any(row["id"] == sweep_id and row["modality"] == modality for row in sweeps)
        rows = await fetch_sweep_items(pool, sweep_id) if mine else None
    except psycopg.Error as exc:
        raise HTTPException(503, "sweeps are unavailable") from exc
    if rows is None:
        raise HTTPException(404, "no such sweep")
    return SweepItemsResponse(items=[SweepItemOut(**row) for row in rows])
