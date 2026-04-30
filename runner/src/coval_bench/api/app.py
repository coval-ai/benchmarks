# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application factory for the Coval Benchmarks API.

Usage::

    from coval_bench.api.app import create_app
    app = create_app()

The factory wires the psycopg3 connection pool lifespan, CORS middleware
(ADR-015 — configured in-app, not infra), slowapi rate-limiting (ADR-013),
and all five routers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from coval_bench.api.ratelimit import _rate_limit_handler, limiter
from coval_bench.api.routers import health, leaderboard, providers, results, runs
from coval_bench.config import Settings, get_settings
from coval_bench.db.conn import lifespan_pool

logger = structlog.get_logger("coval_bench.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional pre-built Settings instance. When ``None``, calls
            ``get_settings()`` which reads from environment variables.

    Returns:
        A fully configured FastAPI app ready to be served by uvicorn.
    """
    resolved: Settings = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("api_startup", runner_sha=resolved.runner_sha)
        async with lifespan_pool(resolved) as pool:
            app.state.pool = pool
            app.state.settings = resolved
            yield
        logger.info("api_shutdown")

    app = FastAPI(
        title="Coval Benchmarks API",
        version=resolved.runner_sha,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # CORS — allowlist read from settings; never hard-coded (ADR-015).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_origin_regex=resolved.cors_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
        max_age=600,
    )

    # Rate limiting (ADR-013) — in-memory per-instance; see ratelimit.py for caveat.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)

    # Routers
    app.include_router(health.router)
    app.include_router(runs.router, prefix="/v1")
    app.include_router(results.router, prefix="/v1")
    app.include_router(leaderboard.router, prefix="/v1")
    app.include_router(providers.router, prefix="/v1")

    return app
