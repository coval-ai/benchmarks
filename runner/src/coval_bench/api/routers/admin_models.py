# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""/v1/admin/models — read and toggle model lifecycle state.

The admin surface behind the benchmarks site's admin page. Both endpoints are
coval-only: the caller must present a Clerk session token whose verified email
is internal (``clerk.internal_email``). Unlike the data endpoints, an active
partner org never narrows the check — a coval dev previewing a partner view can
still administer. Everything else is a 403, valid partner tokens included.

State semantics live in ``coval_bench.db.model_state``; this router only adds
HTTP, auth, and the cache bust that makes a toggle visible to this instance's
other endpoints immediately instead of after the state-cache TTL.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request
from starlette.responses import Response

from coval_bench.api import clerk
from coval_bench.api.deps import MODEL_STATE_CACHE_KEY, get_pool, get_settings
from coval_bench.api.internal import never_shared
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    AdminModelOut,
    AdminModelsResponse,
    ModelStateChangeOut,
    ModelStateOut,
    ModelStatePatch,
)
from coval_bench.config import Settings
from coval_bench.db.model_state import (
    fetch_model_states,
    fetch_recent_history,
    set_model_state,
)
from coval_bench.registries import MODEL_REGISTRY, Benchmark

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["admin"])


def require_internal(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str:
    """The caller's verified coval.dev email, or a 403."""
    email = clerk.internal_email(authorization, settings)
    if email is None:
        raise HTTPException(403, "admin endpoints require a coval identity")
    return email


@router.get("/admin/models", response_model=AdminModelsResponse)
@limiter.limit("60/minute")
async def list_admin_models(
    request: Request,
    response: Response,
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    _email: str = Depends(require_internal),
) -> AdminModelsResponse:
    """Every registry model with its current state and recent history.

    Reads the table directly (not the state cache): the admin page is where
    toggles are made, so it must always show the write it just performed.
    """
    never_shared(response)
    states = await fetch_model_states(pool)
    history = await fetch_recent_history(pool)
    models = [
        AdminModelOut(
            benchmark=m.benchmark.value,
            provider=m.provider,
            model=m.model,
            state=ModelStateOut.model_validate(
                states[(m.benchmark, m.provider, m.model)], from_attributes=True
            ),
            history=[
                ModelStateChangeOut.model_validate(change, from_attributes=True)
                for change in history.get((m.benchmark, m.provider, m.model), [])
            ],
        )
        for m in MODEL_REGISTRY
    ]
    return AdminModelsResponse(models=models)


@router.patch("/admin/models/{benchmark}/{provider}/{model}", response_model=ModelStateOut)
@limiter.limit("30/minute")
async def patch_admin_model(
    request: Request,
    response: Response,
    benchmark: Benchmark,
    provider: str,
    model: str,
    body: ModelStatePatch,
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    email: str = Depends(require_internal),
) -> ModelStateOut:
    """Set one model's state; the change lands in the history table."""
    never_shared(response)
    if not any(
        m.benchmark is benchmark and m.provider == provider and m.model == model
        for m in MODEL_REGISTRY
    ):
        raise HTTPException(404, "unknown model")
    state = await set_model_state(
        pool,
        benchmark=benchmark,
        provider=provider,
        model=model,
        running=body.running,
        shown=body.shown,
        changed_by=email,
    )
    # Bust this instance's state cache so the toggle is immediately visible to
    # every endpoint, not just after the TTL. Other instances converge within
    # the TTL on their own.
    request.app.state.model_state_cache.pop(MODEL_STATE_CACHE_KEY, None)
    logger.info(
        "model_state_changed",
        benchmark=benchmark.value,
        provider=provider,
        model=model,
        running=body.running,
        shown=body.shown,
        changed_by=email,
    )
    return ModelStateOut.model_validate(state, from_attributes=True)
