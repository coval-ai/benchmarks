# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""/v3/models: the model registry. Anyone reads what they may see; coval staff write."""

from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool, get_settings
from coval_bench.api.ratelimit import limiter
from coval_bench.api.routers.admin_models import create_model, etag, update_model
from coval_bench.api.schemas import (
    ErrorResponse,
    ModelCreate,
    ModelOut,
    ModelPatch,
    ModelsResponse,
    ModelUpdateResponse,
)
from coval_bench.api.v3_access import Caller, can_see, coval_only, viewer, writer
from coval_bench.config import Settings
from coval_bench.db.registry_store import ModelRecord, NewModel, RegistryStore
from coval_bench.registries import Benchmark

router = APIRouter(tags=["models"])

_STAFF_FIELDS = ("arena_enabled", "voice", "voices", "updated_at")
_WRITE_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (400, 403, 409)
}


def model_out(record: ModelRecord, who: Caller) -> ModelOut:
    """The model as *who* may see it: staff get how it is run and who last changed it."""
    staff = who.level == "coval"
    row = record.model_dump(exclude=None if staff else set(_STAFF_FIELDS))
    return ModelOut.model_validate(
        {**row, "updated_by_org": record.updated_by_org_id if staff else None}
    )


@router.get("/models", response_model=ModelsResponse, responses={400: {"model": ErrorResponse}})
@limiter.limit("60/minute")
async def get_models(
    request: Request,  # required by slowapi
    modality: Benchmark | None = Query(default=None),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    who: Caller = Depends(viewer),
) -> ModelsResponse:
    """Every model the caller may see, in id order.

    The public sees public models. A company's org also sees that company's
    partner-level models, and coval staff see every model with how it is run and who
    last changed it. ``updated_at``, quoted, is the ``If-Match`` tag a PATCH needs.
    """
    try:
        records = await RegistryStore(pool).list_models()
    except psycopg.Error as exc:
        raise HTTPException(503, "models are unavailable") from exc
    return ModelsResponse(
        models=[
            model_out(record, who)
            for record in sorted(records, key=lambda record: record.id)
            if can_see(who, record.visibility, record.provider)
            and modality in (None, record.modality)
        ]
    )


@router.post("/models", response_model=ModelOut, status_code=201, responses=_WRITE_ERRORS)
@limiter.limit("60/minute")
async def create(
    request: Request,  # required by slowapi
    body: ModelCreate,
    response: Response,
    who: Caller = Depends(coval_only),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    settings: Settings = Depends(get_settings),
) -> ModelOut:
    """Add a model. Coval staff only. The change is recorded against the caller's org."""
    org = writer(who)
    record = await create_model(
        NewModel.model_validate(body.model_dump()),
        RegistryStore(pool),
        settings,
        org_id=org,
    )
    response.headers["ETag"] = etag(record)
    return model_out(record, who)


@router.patch("/models/{model_id}", response_model=ModelUpdateResponse, responses=_WRITE_ERRORS)
@limiter.limit("60/minute")
async def update(
    request: Request,  # required by slowapi
    model_id: int,
    body: ModelPatch,
    response: Response,
    if_match: str | None = Header(default=None),
    who: Caller = Depends(coval_only),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    settings: Settings = Depends(get_settings),
) -> ModelUpdateResponse:
    """Change some of a model's fields. Coval staff only.

    ``If-Match`` must carry the model's last-seen ``ETag``, so an edit made from a stale
    copy is refused. The change is recorded against the caller's org.
    """
    org = writer(who)
    after, warnings = await update_model(model_id, body, if_match, pool, settings, org_id=org)
    response.headers["ETag"] = etag(after)
    return ModelUpdateResponse(model=model_out(after, who), warnings=warnings)
