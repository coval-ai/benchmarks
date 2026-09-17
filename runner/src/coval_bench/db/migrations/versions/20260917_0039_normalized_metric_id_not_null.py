# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S608
"""Enforce normalized metric identities after the hydration release."""

from __future__ import annotations

from alembic import op
from sqlalchemy.engine import Connection

revision = "20260917_0039"
down_revision = "20260917_0038"
branch_labels = None
depends_on = None

_TABLES = (
    "metric_evaluations",
    "dashboard_metric_values",
    "metric_values_by_bucket",
)


def _constraint_exists(bind: Connection, table: str, constraint: str) -> tuple[bool, bool]:
    row = bind.exec_driver_sql(
        """
        SELECT convalidated
        FROM pg_constraint
        WHERE conrelid = %s::regclass AND conname = %s
        """,
        (f"benchmarks_v2.{table}", constraint),
    ).fetchone()
    return row is not None, bool(row[0]) if row is not None else False


def _is_not_null(bind: Connection, table: str) -> bool:
    return bool(
        bind.exec_driver_sql(
            """
            SELECT attnotnull
            FROM pg_attribute
            WHERE attrelid = %s::regclass AND attname = 'metric_id' AND NOT attisdropped
            """,
            (f"benchmarks_v2.{table}",),
        ).scalar()
    )


def _set_lock_timeout(bind: Connection) -> str:
    previous = str(bind.exec_driver_sql("SHOW lock_timeout").scalar())
    bind.exec_driver_sql("SET lock_timeout = '10s'")
    return previous


def _restore_lock_timeout(bind: Connection, previous: str) -> None:
    bind.exec_driver_sql("SELECT set_config('lock_timeout', %s, false)", (previous,))


def upgrade() -> None:
    bind = op.get_bind()
    with op.get_context().autocommit_block():
        previous_timeout = _set_lock_timeout(bind)
        try:
            # Add a migration-owned proof constraint only where the column still
            # needs a scan. This makes a failed upgrade safe to resume at 0038.
            for table in _TABLES:
                check = f"{table}_metric_id_not_null"
                exists, _ = _constraint_exists(bind, table, check)
                if not exists and not _is_not_null(bind, table):
                    bind.exec_driver_sql(
                        f"ALTER TABLE benchmarks_v2.{table} ADD CONSTRAINT {check} "
                        "CHECK (metric_id IS NOT NULL) NOT VALID"
                    )

            # Validate every proof and existing foreign key before acquiring the
            # stronger lock needed by SET NOT NULL.
            for table in _TABLES:
                check = f"{table}_metric_id_not_null"
                exists, validated = _constraint_exists(bind, table, check)
                if exists and not validated:
                    bind.exec_driver_sql(
                        f"ALTER TABLE benchmarks_v2.{table} VALIDATE CONSTRAINT {check}"
                    )
                fk = f"{table}_metric_id_fkey"
                exists, validated = _constraint_exists(bind, table, fk)
                if not exists:
                    raise RuntimeError(f"missing required foreign key constraint: {fk}")
                if exists and not validated:
                    bind.exec_driver_sql(
                        f"ALTER TABLE benchmarks_v2.{table} VALIDATE CONSTRAINT {fk}"
                    )

            for table in _TABLES:
                if not _is_not_null(bind, table):
                    bind.exec_driver_sql(
                        f"ALTER TABLE benchmarks_v2.{table} ALTER COLUMN metric_id SET NOT NULL"
                    )

            # The check constraints have served their purpose; retain the FK.
            for table in _TABLES:
                bind.exec_driver_sql(
                    f"ALTER TABLE benchmarks_v2.{table} DROP CONSTRAINT IF EXISTS "
                    f"{table}_metric_id_not_null"
                )
        finally:
            _restore_lock_timeout(bind, previous_timeout)


def downgrade() -> None:
    """Restore nullable columns while retaining FK validation state."""
    bind = op.get_bind()
    with op.get_context().autocommit_block():
        previous_timeout = _set_lock_timeout(bind)
        try:
            for table in _TABLES:
                bind.exec_driver_sql(
                    f"ALTER TABLE benchmarks_v2.{table} ALTER COLUMN metric_id DROP NOT NULL"
                )
            for table in _TABLES:
                bind.exec_driver_sql(
                    f"ALTER TABLE benchmarks_v2.{table} DROP CONSTRAINT IF EXISTS "
                    f"{table}_metric_id_not_null"
                )
        finally:
            _restore_lock_timeout(bind, previous_timeout)
