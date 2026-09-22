# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit retirement response for the legacy ``GET /v1/results`` reader.

The aggregate and timeline routes under ``/v1/results/*`` remain available.
Row-level normalized evaluations are served by ``GET /v2/results``.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette.requests import Request

from coval_bench.api.ratelimit import limiter

router = APIRouter(tags=["results"])


@router.get(
    "/results",
    status_code=410,
    responses={
        410: {
            "description": "The legacy row-level results reader has been retired.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "The legacy /v1/results endpoint has been retired.",
                        "replacement": "/v2/results",
                    }
                }
            },
        }
    },
)
@limiter.limit("60/minute")
async def retired_results(
    request: Request,
) -> JSONResponse:
    """Return a clear migration response without touching storage or auth state.

    Query strings are intentionally not declared: FastAPI ignores arbitrary
    legacy filters, so even malformed old values still receive 410.
    """
    del request
    return JSONResponse(
        status_code=410,
        content={
            "detail": "The legacy /v1/results endpoint has been retired.",
            "replacement": "/v2/results",
        },
    )
