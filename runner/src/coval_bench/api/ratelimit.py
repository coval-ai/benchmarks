# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""slowapi rate-limit configuration.

The limiter uses in-memory storage by default. This is acceptable for Phase 2:
each Cloud Run instance maintains its own counter, so total throughput could
exceed 60 req/min/IP under multi-instance scaling. If abuse appears, Phase 3
may switch to a Redis-backed limiter (slowapi supports that via ``limits``
storage backends).

Usage::

    from coval_bench.api.ratelimit import limiter, _rate_limit_handler

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_middleware(SlowAPIMiddleware)

All ``/v1/*`` endpoints are decorated with ``@limiter.limit("60/minute")``.
``/healthz`` and ``/readyz`` are exempt (they are polled by GCP load balancers).
"""

from __future__ import annotations

from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

limiter: Limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],  # no global default — only /v1/* routes carry explicit limits
)


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a 429 JSON response when the rate limit is exceeded."""
    return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
