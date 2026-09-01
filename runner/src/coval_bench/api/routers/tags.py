# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The tag vocabulary: public to read, coval-gated to extend.

The site reads it to build per-table filters; the values are already public
through ``/v1/providers``, so the read takes no token.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from psycopg_pool import AsyncConnectionPool

from coval_bench.api.deps import get_pool, require_coval_admin
from coval_bench.api.schemas import TagOut, TagsResponse
from coval_bench.db.registry_store import DuplicateKey, RegistryStore, TagRecord

router = APIRouter(tags=["tags"])


@router.get("/tags", response_model=TagsResponse)
async def list_tags(
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> TagsResponse:
    """The FEATURES tag vocabulary, ordered by value."""
    records = await RegistryStore(pool).list_tags()
    return TagsResponse(tags=[TagOut.model_validate(record.model_dump()) for record in records])


@router.post(
    "/tags",
    response_model=TagOut,
    status_code=201,
    dependencies=[Depends(require_coval_admin)],
)
async def create_tag(
    body: TagOut,
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> TagOut:
    """Add a vocabulary entry. There is no delete: models reference tags."""
    try:
        await RegistryStore(pool).insert_tag(TagRecord.model_validate(body.model_dump()))
    except DuplicateKey as exc:
        raise HTTPException(409, str(exc)) from exc
    return body
