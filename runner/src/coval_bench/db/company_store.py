# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``CompanyStore``: reads and admin writes of companies and their Clerk org."""

from __future__ import annotations

from datetime import datetime

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from coval_bench.db.benchmark_store import RegistryConflict

CompanyPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]

_COLUMNS = (
    "slug, name, clerk_org_id AS org_id, updated_by_user_id, updated_by_org_id,"
    " updated_by_email, updated_at"
)
_SET_ORG = f"""
    UPDATE benchmarks_v2.companies
    SET clerk_org_id = %s, updated_by_user_id = %s, updated_by_org_id = %s,
        updated_by_email = %s, updated_at = now()
    WHERE slug = %s
    RETURNING {_COLUMNS}
"""  # noqa: S608 — constant columns


class Company(BaseModel):
    slug: str
    name: str
    org_id: str | None
    updated_by_user_id: str | None
    updated_by_org_id: str | None = None
    updated_by_email: str | None
    updated_at: datetime


class CompanyStore:
    """Per-pool persistence helper for companies."""

    def __init__(self, pool: CompanyPool) -> None:
        self._pool = pool

    async def companies(self) -> list[Company]:
        sql = f"SELECT {_COLUMNS} FROM benchmarks_v2.companies ORDER BY slug"  # noqa: S608
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql)).fetchall()
        return [Company.model_validate(dict(row)) for row in rows]

    async def create(
        self,
        *,
        slug: str,
        name: str,
        user_id: str | None,
        email: str | None,
        by_org: str | None = None,
    ) -> Company:
        sql = f"""
            INSERT INTO benchmarks_v2.companies
                (slug, name, updated_by_user_id, updated_by_org_id, updated_by_email)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING {_COLUMNS}
        """  # noqa: S608 — constant columns
        try:
            async with self._pool.connection() as conn:
                row = await (
                    await conn.execute(sql, (slug, name, user_id, by_org, email))
                ).fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise RegistryConflict(f"company {slug!r} already exists") from exc
        return Company.model_validate(dict(row or {}))

    async def set_org(
        self,
        slug: str,
        org_id: str | None,
        *,
        user_id: str | None,
        email: str | None,
        by_org: str | None = None,
    ) -> Company | None:
        """Set or clear the company's Clerk org; None when there is no such company.

        An org that already belongs to another company is a conflict.
        """
        try:
            async with self._pool.connection() as conn:
                row = await (
                    await conn.execute(_SET_ORG, (org_id, user_id, by_org, email, slug))
                ).fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise RegistryConflict("that org belongs to another company") from exc
        return None if row is None else Company.model_validate(dict(row))
