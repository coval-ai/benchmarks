# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Click commands for database management.

Registered on the ``db`` group in ``coval_bench.__main__``.

Commands
--------
migrate   Run ``alembic upgrade head``. Idempotent. Executed at Cloud Run
          Job boot before the benchmark run starts.
db-check  Open a connection, run ``SELECT 1``, print OK and exit 0.
          Used as a liveness probe in CI and Cloud Run health checks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click


@click.command(name="migrate")
def db_migrate() -> None:
    """Run ``alembic upgrade head``. Idempotent."""
    from alembic import command
    from alembic.config import Config

    from coval_bench.config import get_settings

    # alembic.ini lives at the same level as pyproject.toml (runner root).
    ini_path = Path(__file__).parents[4] / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("sqlalchemy.url", str(get_settings().database_url))
    command.upgrade(cfg, "head")
    click.echo("alembic upgrade head: done")


@click.command(name="db-check")
def db_check() -> None:
    """Open a connection, run ``SELECT 1``, exit 0 on success."""
    import psycopg

    from coval_bench.config import get_settings

    settings = get_settings()

    async def _check() -> None:
        async with await psycopg.AsyncConnection.connect(str(settings.database_url)) as conn:
            cur = await conn.execute("SELECT 1")
            row = await cur.fetchone()
            if row is None or row[0] != 1:  # pragma: no cover
                raise RuntimeError("SELECT 1 returned unexpected result")

    asyncio.run(_check())
    click.echo("db-check: OK")
