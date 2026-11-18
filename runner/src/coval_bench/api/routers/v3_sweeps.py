# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v3/sweeps: how far along each quality sweep is. Coval staff only."""

from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ErrorResponse, SweepOut, SweepsResponse
from coval_bench.api.v3_access import coval_only
from coval_bench.db.sweep_store import fetch_sweeps

router = APIRouter(tags=["sweeps"], dependencies=[Depends(coval_only)])


@router.get("/sweeps", response_model=SweepsResponse, responses={403: {"model": ErrorResponse}})
@limiter.limit("60/minute")
async def get_sweeps(
    request: Request,  # required by slowapi
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> SweepsResponse:
    """One row per sweep, oldest first, finished or not.

    ``result_count`` items have run, ``queued_count`` are still to run and
    ``gave_up_count`` ran out of tries. A sweep with ``completed_at`` set has nothing
    left to run. A model has a sweep per dataset for each of its versions, so an old
    version's sweeps are listed too.
    """
    try:
        rows = await fetch_sweeps(pool)
    except psycopg.Error as exc:
        raise HTTPException(503, "sweeps are unavailable") from exc
    return SweepsResponse(sweeps=[SweepOut(**row) for row in rows])
