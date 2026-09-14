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

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "as_of": self.as_of,
            "published_at": self.published_at,
            "definition_revision": self.definition_revision,
            "stale": self.stale,
        }


async def require_snapshot(conn: AsyncConnection[Any]) -> Snapshot:
    """Validate state and definition identity inside the caller's transaction."""
    try:
        result = await conn.execute(
            """SELECT generation,as_of,published_at,definition_revision,definition_fingerprint
               FROM benchmarks_v2.dashboard_summary_state WHERE id=true"""
        )
    except UndefinedTable as exc:
        raise HTTPException(status_code=503, detail="dashboard_snapshot_not_ready") from exc
    row = await result.fetchone()
    if row is None:
        raise HTTPException(status_code=503, detail="dashboard_snapshot_not_ready")
    generation = row["generation"] if isinstance(row, dict) else row[0]
    as_of = row["as_of"] if isinstance(row, dict) else row[1]
    published_at = row["published_at"] if isinstance(row, dict) else row[2]
    revision = row["definition_revision"] if isinstance(row, dict) else row[3]
    fingerprint = row["definition_fingerprint"] if isinstance(row, dict) else row[4]
    if generation == 0 or as_of is None or published_at is None:
        raise HTTPException(status_code=503, detail="dashboard_snapshot_not_ready")
    if revision != DEFINITION_REVISION or fingerprint != aggregation_fingerprint():
        raise HTTPException(status_code=503, detail="dashboard_snapshot_not_ready")
    now = dt.datetime.now(dt.UTC)
    stale = now - min(as_of, published_at) > dt.timedelta(minutes=30)
    return Snapshot(int(generation), as_of, published_at, int(revision), stale)


@asynccontextmanager
async def dashboard_read(
    pool: AsyncConnectionPool[Any], *, saved: bool
) -> AsyncIterator[AsyncConnection[Any]]:
    """Read saved state and rows in one transaction, including autocommit pools."""
    try:
        async with pool.connection() as conn, conn.transaction():
            conn.row_factory = dict_row
            if saved:
                await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            yield conn
    except (UndefinedTable, ObjectNotInPrerequisiteState) as exc:
        if not saved:
            raise
        raise HTTPException(status_code=503, detail="dashboard_snapshot_not_ready") from exc
