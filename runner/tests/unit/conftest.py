# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for unit tests."""

from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import psycopg
import psycopg.rows
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql_proc

# One embedded Postgres server (random free port) for all DB-backed unit
# tests; each client fixture still gets a clean per-test database.
pg_proc = postgresql_proc(port=None)

# ---------------------------------------------------------------------------
# Postgres plumbing shared by DB-backed unit tests. Kept here because each test
# module otherwise rebuilds the same DSN, alembic config and pool by hand.
# ---------------------------------------------------------------------------

_INI_PATH = Path(__file__).parents[2] / "alembic.ini"


def async_dsn(conn: psycopg.Connection[Any]) -> str:
    """A libpq URL for the embedded server behind *conn*."""
    info = conn.info
    user, password = info.user or "", info.password or ""
    credentials = f"{quote_plus(user)}:{quote_plus(password)}" if password else quote_plus(user)
    return (
        f"postgresql://{credentials}@{info.host or 'localhost'}:"
        f"{info.port or 5432}/{info.dbname or 'test'}"
    )


def apply_migrations(conn: psycopg.Connection[Any]) -> None:
    """Bring the per-test database up to head."""
    cfg = AlembicConfig(str(_INI_PATH))
    cfg.set_main_option(
        "sqlalchemy.url", async_dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    alembic_command.upgrade(cfg, "head")


async def open_pool(conn: psycopg.Connection[Any]) -> AsyncConnectionPool[Any]:
    """An open pool onto *conn*'s database, configured like the app's."""
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=async_dsn(conn),
        min_size=1,
        max_size=2,
        open=False,
        kwargs={"autocommit": True, "row_factory": psycopg.rows.dict_row},
    )
    await pool.open()
    return pool
