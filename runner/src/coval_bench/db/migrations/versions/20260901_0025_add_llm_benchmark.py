# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Embedded SQL is formatted as executable migration text.

"""Add the LLM benchmark: widen the modality CHECKs and seed the Phonely entry.

Revision ID: 20260901_0025
Revises:     20260831_0024
Create Date: 2026-09-01

Every table that stores a benchmark or modality string carries an inline
column-level CHECK, which Postgres names deterministically
``<table>_<column>_check``. Adding the LLM benchmark widens all six to include
``'LLM'``. The per-window matviews derive from ``results`` and carry no
constraint, so they need no change.

Unlike 20260707_0009, this runs statement by statement in autocommit rather
than as one transaction. The benchmark tables are never idle: half-hourly
runner batches overlap, dual-write inserts observations continuously, and each
run's rollup refresh reads the normalized tables for seconds at a time. A
single transaction that widened one table and then waited for the next either
deadlocked against an API request (three attempts on 2026-09-02) or stalled
every reader behind locks it already held. Here each swap is its own instant
statement (``NOT VALID`` skips the scan), a lock timeout bounds every wait, a
timed-out statement is retried, and validation runs last under a lock that
blocks neither readers nor writers. Every statement is idempotent, so a partial
run is completed by simply running the migration again.

The Phonely text agent is the first LLM entry. It is seeded here rather than in
the code registry because the database is authoritative for the roster.
Collected but unpublished: its results stay behind the early-access grant.
"""

from __future__ import annotations

import time

import psycopg.errors
import sqlalchemy.exc
from alembic import op

revision = "20260901_0025"
down_revision = "20260831_0024"
branch_labels = None
depends_on = None

_CHECKS: tuple[tuple[str, str], ...] = (
    ("models", "modality"),
    ("model_history", "modality"),
    ("benchmark_observations", "benchmark"),
    ("metric_values_by_bucket", "benchmark"),
    ("results_by_bucket", "benchmark"),
    ("results", "benchmark"),
)

_WIDE = "'STT','TTS','S2S','LLM'"
_NARROW = "'STT','TTS','S2S'"

_LOCK_TIMEOUT = "10s"
_LOCK_ATTEMPTS = 12
_RETRY_PAUSE_SECONDS = 5.0


def _execute_retrying_locks(sql: str) -> None:
    for attempt in range(1, _LOCK_ATTEMPTS + 1):
        try:
            op.execute(sql)
            return
        except sqlalchemy.exc.OperationalError as exc:
            lock_timeout = isinstance(exc.orig, psycopg.errors.LockNotAvailable)
            if not lock_timeout or attempt == _LOCK_ATTEMPTS:
                raise
            time.sleep(_RETRY_PAUSE_SECONDS)


def upgrade() -> None:
    """Admit 'LLM' everywhere a benchmark is stored, then seed the Phonely row."""
    with op.get_context().autocommit_block():
        op.execute(f"SET lock_timeout = '{_LOCK_TIMEOUT}'")
        for table, column in _CHECKS:
            name = f"{table}_{column}_check"
            _execute_retrying_locks(
                f"ALTER TABLE benchmarks_v2.{table} DROP CONSTRAINT IF EXISTS {name}"
            )
            _execute_retrying_locks(
                f"ALTER TABLE benchmarks_v2.{table} ADD CONSTRAINT {name} CHECK ({column} IN ({_WIDE})) NOT VALID"
            )
        op.get_bind().exec_driver_sql(
            """
            INSERT INTO benchmarks_v2.models
                (modality, provider, model, voice, voices, creator, source, licensing,
                 on_prem, region, arena_enabled, collected, published, updated_by_user_id)
            SELECT 'LLM', 'phonely', 'phonely-agent', NULL, '[]'::jsonb, NULL, 'official-api', 'proprietary',
                   FALSE, 'us', FALSE, TRUE, FALSE, 'migration:20260901_0025'
            WHERE NOT EXISTS (
                SELECT 1 FROM benchmarks_v2.models
                WHERE modality = 'LLM' AND provider = 'phonely' AND model = 'phonely-agent'
            );
            """
        )
        for table, column in _CHECKS:
            _execute_retrying_locks(
                f"ALTER TABLE benchmarks_v2.{table} VALIDATE CONSTRAINT {table}_{column}_check"
            )
        op.execute("RESET lock_timeout")


def downgrade() -> None:
    """Remove the seed row unless edited since, then re-narrow (fails if LLM rows remain)."""
    op.get_bind().exec_driver_sql(
        """
        DELETE FROM benchmarks_v2.models
        WHERE updated_by_user_id = 'migration:20260901_0025'
          AND modality = 'LLM' AND provider = 'phonely' AND model = 'phonely-agent';
        """
    )
    # Transactional and validated inline: a narrowing that would strand LLM rows
    # must fail atomically rather than leave half the tables narrowed.
    op.execute(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT}'")
    for table, column in _CHECKS:
        name = f"{table}_{column}_check"
        op.execute(f"ALTER TABLE benchmarks_v2.{table} DROP CONSTRAINT IF EXISTS {name}")  # noqa: S608
        op.execute(
            f"ALTER TABLE benchmarks_v2.{table} ADD CONSTRAINT {name} CHECK ({column} IN ({_NARROW}))"  # noqa: S608
        )
