# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Alembic environment — online-only sync psycopg3 migrations.

The database URL is read from the ``DATABASE_URL`` environment variable
(not from ``alembic.ini``).  The ``sqlalchemy.url`` config option is also
accepted when set programmatically via ``cfg.set_main_option`` (as done by
``coval_bench.db.cli.db_migrate``).

Offline migrations are not supported; the runner always has a live DB
connection (Cloud SQL Auth Proxy).

Notes
-----
- DB roles and GRANTs are managed by Terraform, not Alembic.
- The ``results_24h`` materialized view is refreshed by a Cloud Scheduler
  cron; Alembic only creates the initial definition.
- ``alembic_version`` is stored in the public schema (default) so that
  Alembic can create it before ``benchmarks_v2`` is created.
- We connect via psycopg3 sync connection + SQLAlchemy ``create_engine``
  (not the async variant) to avoid a ``greenlet`` dependency.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None  # no ORM — raw SQL only


def _get_url() -> str:
    """Resolve the database URL.

    Priority:
    1. ``sqlalchemy.url`` set programmatically (e.g. from ``db_migrate`` CLI).
    2. ``DATABASE_URL`` environment variable.

    The URL is normalised to use the ``postgresql+psycopg://`` driver so that
    SQLAlchemy's sync psycopg3 dialect is used (no greenlet required).
    """
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set and "
            "sqlalchemy.url was not configured programmatically."
        )
    # Normalise to sync psycopg3 driver
    url = url.replace("postgresql+psycopg2://", "postgresql+psycopg://")
    if not url.startswith("postgresql+"):
        url = url.replace("postgresql://", "postgresql+psycopg://")
    return url


def run_migrations_online() -> None:
    """Run migrations against a live database (sync psycopg3 via SQLAlchemy).

    The ``alembic_version`` table is stored in the default (public) schema so
    that Alembic can create it before the ``benchmarks_v2`` schema exists.
    Migration scripts are responsible for setting the search_path themselves
    (via ``op.execute("SET search_path TO benchmarks_v2")``).
    """
    engine = create_engine(_get_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            # Store alembic_version in public schema (exists before migration)
            version_table_schema="public",
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
