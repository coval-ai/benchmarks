# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Async psycopg3 connection pool, shared by the runner and the API.

Usage (runner orchestrator / FastAPI lifespan)::

    async with lifespan_pool(settings) as pool:
        writer = RunWriter(pool)
        ...

Notes
-----
- ``max_size=4``: conservative for the runner (single-threaded async) and the
  API (min=0/max=5 Cloud Run instances each with their own pool).  Cloud SQL
  ``db-custom-1-3840`` allows ~50 connections; this stays well under.
- ``open=False``: explicit open is required for psycopg3 ``AsyncConnectionPool``.
  Never rely on lazy open — wire ``.open()`` and ``.close()`` into your startup /
  shutdown hook (FastAPI lifespan or runner orchestrator).
- ``autocommit=False``: explicit transaction control for ``RunWriter``.
- ``row_factory=dict_row``: rows returned as dicts; ``RunWriter`` uses
  ``Model.model_validate(row)`` to convert back to typed objects.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

if TYPE_CHECKING:
    from coval_bench.config import Settings

_pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]] | None = None


async def get_pool(
    settings: Settings,
) -> AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]:
    """Return the process-singleton async connection pool.

    The pool is created once per process (first call wins).  Callers are
    responsible for calling ``.open()`` before use and ``.close()`` at shutdown.
    Prefer ``lifespan_pool`` for FastAPI; use this directly in the runner
    orchestrator's startup/shutdown hooks.
    """
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=str(settings.database_url),
            min_size=1,
            max_size=4,
            open=False,  # explicit open in caller's startup
            kwargs={
                "autocommit": False,
                "row_factory": psycopg.rows.dict_row,
            },
        )
    return _pool


@asynccontextmanager
async def lifespan_pool(
    settings: Settings,
) -> AsyncIterator[AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]]:
    """Async context manager suitable for FastAPI's lifespan handler.

    Opens the pool on enter, closes it on exit (even on exception).

    Example (FastAPI)::

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            async with lifespan_pool(get_settings()) as pool:
                app.state.pool = pool
                yield
    """
    pool = await get_pool(settings)
    await pool.open()
    try:
        yield pool
    finally:
        await pool.close()
