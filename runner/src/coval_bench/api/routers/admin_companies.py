# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Admin company endpoints: companies and the Clerk org each one has.

Coval-org only (``require_coval_admin``) and every response is private. A company has
at most one org and an org belongs to one company. There is no DELETE for a company.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Response
from psycopg_pool import AsyncConnectionPool

from coval_bench.api.clerk import CovalAdmin
from coval_bench.api.deps import get_pool, require_coval_admin
from coval_bench.api.internal import never_shared
from coval_bench.api.schemas import (
    AdminCompaniesResponse,
    AdminCompanyCreate,
    AdminCompanyOrgPut,
)
from coval_bench.db.benchmark_store import RegistryConflict
from coval_bench.db.company_store import Company, CompanyStore

router = APIRouter(tags=["admin"], dependencies=[Depends(require_coval_admin)])


async def _guarded[T](call: Awaitable[T]) -> T:
    """Run a store call; a missing table or grant is a 503, a broken rule a 409."""
    try:
        return await call
    except RegistryConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(503, "companies are unavailable") from exc


def _found(company: Company | None) -> Company:
    if company is None:
        raise HTTPException(404, "no such company")
    return company


@router.get("/admin/companies", response_model=AdminCompaniesResponse)
async def list_admin_companies(
    response: Response, pool: AsyncConnectionPool[Any] = Depends(get_pool)
) -> AdminCompaniesResponse:
    """Every company with its Clerk org, if it has one."""
    never_shared(response)
    return AdminCompaniesResponse(companies=await _guarded(CompanyStore(pool).companies()))


@router.post("/admin/companies", response_model=Company, status_code=201)
async def create_admin_company(
    body: AdminCompanyCreate,
    response: Response,
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> Company:
    """Add a company. Its slug is the provider string its models carry."""
    never_shared(response)
    return await _guarded(
        CompanyStore(pool).create(
            slug=body.slug, name=body.name, user_id=admin.user_id, email=admin.email
        )
    )


@router.put("/admin/companies/{slug}/org", response_model=Company)
async def set_admin_company_org(
    slug: str,
    body: AdminCompanyOrgPut,
    response: Response,
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> Company:
    """Set the company's Clerk org, so that org sees the company's unpublished models."""
    never_shared(response)
    return _found(
        await _guarded(
            CompanyStore(pool).set_org(slug, body.org_id, user_id=admin.user_id, email=admin.email)
        )
    )


@router.delete("/admin/companies/{slug}/org", response_model=Company)
async def clear_admin_company_org(
    slug: str,
    response: Response,
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> Company:
    """Remove the company's Clerk org."""
    never_shared(response)
    return _found(
        await _guarded(
            CompanyStore(pool).set_org(slug, None, user_id=admin.user_id, email=admin.email)
        )
    )
