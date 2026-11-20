# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""/v3/companies: who makes and serves the models. Anyone reads; coval staff write."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    AdminCompanyCreate,
    AdminCompanyOrgPut,
    CompaniesResponse,
    CompanyOut,
    ErrorResponse,
)
from coval_bench.api.v3_access import Caller, can_see, coval_only, viewer, writer
from coval_bench.db.benchmark_store import RegistryConflict
from coval_bench.db.company_store import Company, CompanyStore
from coval_bench.db.registry_store import RegistryStore

router = APIRouter(tags=["companies"])

_WRITE_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (400, 403, 404, 409)
}


async def _guarded[T](call: Awaitable[T]) -> T:
    try:
        return await call
    except RegistryConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(503, "companies are unavailable") from exc


def _out(company: Company | None, who: Caller) -> CompanyOut:
    if company is None:
        raise HTTPException(404, "no such company")
    if who.level != "coval":
        return CompanyOut(slug=company.slug, name=company.name)
    return CompanyOut(
        slug=company.slug,
        name=company.name,
        org_id=company.org_id,
        updated_by_org=company.updated_by_org_id,
        updated_at=company.updated_at,
    )


@router.get("/companies", response_model=CompaniesResponse)
@limiter.limit("60/minute")
async def get_companies(
    request: Request,  # required by slowapi
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    who: Caller = Depends(viewer),
) -> CompaniesResponse:
    """The companies behind the models the caller may see, in slug order.

    A company is listed when it serves or makes a model the caller sees, so one whose
    models are all unpublished stays unlisted. A company's org always sees its own, and
    coval staff see every company with its Clerk org and who last changed it.
    """
    companies = await _guarded(CompanyStore(pool).companies())
    if who.level != "coval":
        models = await _guarded(RegistryStore(pool).list_models())
        shown = {who.company}
        for model in models:
            if can_see(who, model.visibility, model.provider):
                shown.update((model.provider, model.creator))
        companies = [company for company in companies if company.slug in shown]
    return CompaniesResponse(companies=[_out(company, who) for company in companies])


@router.post("/companies", response_model=CompanyOut, status_code=201, responses=_WRITE_ERRORS)
@limiter.limit("60/minute")
async def create(
    request: Request,  # required by slowapi
    body: AdminCompanyCreate,
    who: Caller = Depends(coval_only),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> CompanyOut:
    """Add a company. Coval staff only. Its slug is the provider string its models carry."""
    company = await _guarded(
        CompanyStore(pool).create(
            slug=body.slug, name=body.name, user_id=None, email=None, by_org=writer(who)
        )
    )
    return _out(company, who)


@router.put("/companies/{slug}/org", response_model=CompanyOut, responses=_WRITE_ERRORS)
@limiter.limit("60/minute")
async def set_org(
    request: Request,  # required by slowapi
    slug: str,
    body: AdminCompanyOrgPut,
    who: Caller = Depends(coval_only),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> CompanyOut:
    """Set the company's Clerk org, so that org sees the company's unpublished models."""
    company = await _guarded(
        CompanyStore(pool).set_org(slug, body.org_id, user_id=None, email=None, by_org=writer(who))
    )
    return _out(company, who)


@router.delete("/companies/{slug}/org", response_model=CompanyOut, responses=_WRITE_ERRORS)
@limiter.limit("60/minute")
async def clear_org(
    request: Request,  # required by slowapi
    slug: str,
    who: Caller = Depends(coval_only),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> CompanyOut:
    """Remove the company's Clerk org. Coval staff only."""
    company = await _guarded(
        CompanyStore(pool).set_org(slug, None, user_id=None, email=None, by_org=writer(who))
    )
    return _out(company, who)
