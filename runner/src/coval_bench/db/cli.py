# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Click commands for database management.

Registered on the ``db`` group in ``coval_bench.__main__``.

Commands
--------
migrate   Upgrade to head, or to an explicitly selected target.
db-check  Open a connection, run ``SELECT 1``, print OK and exit 0.
          Used as a liveness probe in CI and Cloud Run health checks.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click
from alembic import command
from alembic.config import Config


@click.command(name="migrate")
@click.option("--revision", type=str, help="Explicit Alembic target instead of head.")
def db_migrate(revision: str | None) -> None:
    """Upgrade to head, or to an explicitly requested target."""
    from coval_bench.config import get_settings

    # alembic.ini lives at the same level as pyproject.toml (runner root):
    # src/coval_bench/db/cli.py → parents[3] resolves to runner/.
    ini_path = Path(__file__).parents[3] / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("sqlalchemy.url", str(get_settings().database_url))
    target = revision or "head"
    command.upgrade(cfg, target)
    click.echo(f"alembic upgrade {target}: done")


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


def _timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise click.BadParameter("use an ISO timestamp with timezone") from exc
    if parsed.tzinfo is None:
        raise click.BadParameter("timestamp must include a timezone")
    return parsed


def _json_default(value: object) -> str:
    """Encode datetimes in maintenance results as JSON-safe ISO strings."""
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


@click.command(name="refresh-dashboard-aggregates")
@click.option("--as-of", type=str, help="Fixed UTC snapshot boundary for validation or replay.")
def refresh_dashboard_aggregates(as_of: str | None) -> None:
    """Reconcile source/hour repairs and publish rolling summary snapshots."""
    from coval_bench.config import get_settings
    from coval_bench.db.conn import lifespan_pool
    from coval_bench.db.dashboard_aggregates import refresh_dashboard_aggregates as maintain

    at = _timestamp(as_of)

    async def refresh() -> None:
        async with lifespan_pool(get_settings()) as pool:
            result = await maintain(pool, as_of=at)
            click.echo(json.dumps(asdict(result), default=_json_default))

    asyncio.run(refresh())
