# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Click commands for database management.

Registered on the ``db`` group in ``coval_bench.__main__``.

Commands
--------
migrate   Apply the compatibility revision at boot, or an explicitly selected target.
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
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from sqlalchemy import create_engine

_COMPATIBILITY_REVISION = "20261005_0043"


def _default_migration_target(cfg: Config, database_url: str) -> str | None:
    """Resolve the compatibility target from the installed Alembic graph."""
    script = ScriptDirectory.from_config(cfg)
    cap = script.get_revision(_COMPATIBILITY_REVISION)
    if cap is None:
        raise click.ClickException(f"compatibility revision not found: {_COMPATIBILITY_REVISION}")
    url = database_url.replace("postgresql+psycopg2://", "postgresql+psycopg://")
    url = url.replace("postgresql://", "postgresql+psycopg://")
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            heads = tuple(MigrationContext.configure(connection).get_current_heads())
    finally:
        engine.dispose()
    if len(heads) > 1:
        raise click.ClickException("database has multiple or divergent Alembic revisions")
    if not heads:
        return _COMPATIBILITY_REVISION
    try:
        current = script.get_revision(heads[0])
    except CommandError as exc:
        raise click.ClickException(f"unknown Alembic revision: {heads[0]}") from exc
    if current is None:
        raise click.ClickException(f"unknown Alembic revision: {heads[0]}")
    current_ancestors = {
        node.revision for node in script.iterate_revisions(current.revision, "base")
    }
    if cap.revision in current_ancestors:
        return None
    cap_ancestors = {node.revision for node in script.iterate_revisions(cap.revision, "base")}
    if current.revision in cap_ancestors:
        return _COMPATIBILITY_REVISION
    raise click.ClickException(
        f"database revision {heads[0]} is not the compatibility revision or its descendant"
    )


@click.command(name="migrate")
@click.option("--revision", type=str, help="Explicit Alembic target for staged cleanup.")
def db_migrate(revision: str | None) -> None:
    """Apply the compatibility cap or an explicitly requested target."""
    from coval_bench.config import get_settings

    # alembic.ini lives at the same level as pyproject.toml (runner root):
    # src/coval_bench/db/cli.py → parents[3] resolves to runner/.
    ini_path = Path(__file__).parents[3] / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("sqlalchemy.url", str(get_settings().database_url))
    if revision is not None:
        cfg.attributes["allow_metric_code_cleanup"] = True
        command.upgrade(cfg, revision)
        click.echo(f"alembic upgrade {revision}: done")
        return
    target = _default_migration_target(cfg, str(get_settings().database_url))
    if target is None:
        click.echo(f"alembic upgrade {_COMPATIBILITY_REVISION}: already applied")
        return
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
    from coval_bench.db.dashboard_aggregates import reconcile_dashboard_aggregates

    at = _timestamp(as_of)

    async def refresh() -> None:
        async with lifespan_pool(get_settings()) as pool:
            result = await reconcile_dashboard_aggregates(pool, as_of=at)
            click.echo(json.dumps(asdict(result), default=_json_default))

    asyncio.run(refresh())
