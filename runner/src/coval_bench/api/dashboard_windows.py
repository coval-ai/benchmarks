"""Readiness and freshness checks for atomically published dashboard snapshots."""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from psycopg import AsyncConnection
from psycopg.errors import ObjectNotInPrerequisiteState, UndefinedTable
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.dashboard_contracts import DEFINITION_REVISION, aggregation_fingerprint


@dataclass(frozen=True)
class Snapshot:
    generation: int
    as_of: dt.datetime
    published_at: dt.datetime
    definition_revision: int
    stale: bool


async def require_window_state(conn: AsyncConnection[Any]) -> Snapshot:
    """Validate state and definition identity inside the caller's transaction."""
    result = await conn.execute(
        """SELECT generation,as_of,published_at,definition_revision,definition_fingerprint
           FROM benchmarks_v2.dashboard_window_state WHERE id=true"""
    )
    row = await result.fetchone()
    if (
        row is None
        or row["generation"] == 0
        or row["as_of"] is None
        or row["published_at"] is None
        or row["definition_revision"] != DEFINITION_REVISION
        or row["definition_fingerprint"] != aggregation_fingerprint()
    ):
        raise HTTPException(status_code=503, detail="dashboard_snapshot_not_ready")
    as_of, published_at = row["as_of"], row["published_at"]
    # Allow two hourly maintenance intervals before warning about stale data.
    stale = dt.datetime.now(dt.UTC) - min(as_of, published_at) > dt.timedelta(hours=2)
    return Snapshot(
        int(row["generation"]), as_of, published_at, int(row["definition_revision"]), stale
    )


@asynccontextmanager
async def dashboard_read(pool: AsyncConnectionPool[Any]) -> AsyncIterator[AsyncConnection[Any]]:
    """Read saved state and rows in one transaction, including autocommit pools."""
    try:
        async with pool.connection() as conn, conn.transaction():
            conn.row_factory = dict_row
            await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            yield conn
    except (UndefinedTable, ObjectNotInPrerequisiteState) as exc:
        raise HTTPException(status_code=503, detail="dashboard_snapshot_not_ready") from exc
