# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Admin health endpoint: which models the latency probe can't reach, and why.

It reads what the probe already stored and calls no provider, so it is as fresh
as the last probe tick. Coval-org only, and never cached.
"""

from __future__ import annotations

from typing import Any, Literal

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Response
from psycopg_pool import AsyncConnectionPool

from coval_bench.api.deps import get_pool, require_coval_admin
from coval_bench.api.internal import never_shared
from coval_bench.api.schemas import (
    AdminHealthResponse,
    AdminModelHealthOut,
    AdminProbeFailureOut,
)
from coval_bench.arena.provider_health import classify_failure
from coval_bench.db.probe_store import fetch_health

router = APIRouter(tags=["admin"], dependencies=[Depends(require_coval_admin)])

Kind = Literal["auth", "credit", "rate_limit", "timeout", "other"]


def _kind(status_code: int | None, error: str) -> Kind:
    reason = classify_failure(status_code, error)
    if reason is not None:
        return reason.value
    return "timeout" if error.startswith("timed out") else "other"


def _model_out(row: dict[str, Any]) -> AdminModelHealthOut:
    failure = None
    if row["failed_at"] is not None:
        failure = AdminProbeFailureOut(
            at=row["failed_at"],
            kind=_kind(row["status_code"], row["error"]),
            origin=row["failure_origin"],
            status_code=row["status_code"],
            error=row["error"],
        )
    return AdminModelHealthOut(
        model_id=row["model_id"],
        modality=row["modality"],
        provider=row["provider"],
        model=row["model"],
        healthy=row["last_success_at"] == row["last_probe_at"],
        probe_count=row["probe_count"],
        failed_count=row["failed_count"],
        last_probe_at=row["last_probe_at"],
        last_success_at=row["last_success_at"],
        last_failure=failure,
    )


@router.get("/admin/health", response_model=AdminHealthResponse)
async def admin_health(
    response: Response, pool: AsyncConnectionPool[Any] = Depends(get_pool)
) -> AdminHealthResponse:
    """Every model probed in the last 24 hours, the ones failing most first."""
    never_shared(response)
    try:
        rows = await fetch_health(pool)
    except psycopg.Error as exc:
        raise HTTPException(503, "probe results are unavailable") from exc
    return AdminHealthResponse(models=[_model_out(row) for row in rows])
