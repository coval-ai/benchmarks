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

Unlike 20260707_0009, the constraints are re-added ``NOT VALID`` and validated
afterwards outside the transaction. A plain ``ADD CONSTRAINT`` scans the table
under an ACCESS EXCLUSIVE lock; on ``results`` that scan is long enough for
every concurrent reader to queue behind it, and a queued API request that
already holds ``results_by_bucket`` deadlocks the migration's next statement.
``NOT VALID`` makes each swap instantaneous, ``VALIDATE CONSTRAINT`` takes only
SHARE UPDATE EXCLUSIVE, and ``results`` goes last so nothing is held while
waiting for the rollup table. ``lock_timeout`` turns any remaining contention
into a fast, clean failure that the job can simply retry.

The Phonely text agent is the first LLM entry. It is seeded here rather than in
the code registry because the database is authoritative for the roster.
Collected but unpublished: its results stay behind the early-access grant.
"""

from __future__ import annotations

from alembic import op

revision = "20260901_0025"
down_revision = "20260831_0024"
branch_labels = None
depends_on = None

# results last: the concurrent pattern is "hold results_by_bucket, then read results".
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


def _swap_checks(allowed: str, *, validate_inline: bool) -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    suffix = "" if validate_inline else " NOT VALID"
    for table, column in _CHECKS:
        name = f"{table}_{column}_check"
        op.execute(f"ALTER TABLE benchmarks_v2.{table} DROP CONSTRAINT IF EXISTS {name}")  # noqa: S608
        op.execute(
            f"ALTER TABLE benchmarks_v2.{table} ADD CONSTRAINT {name} CHECK ({column} IN ({allowed})){suffix}"  # noqa: S608
        )


def _validate_checks() -> None:
    # Outside the migration transaction so the ACCESS EXCLUSIVE locks from the
    # swaps are released before the scans start; VALIDATE blocks nobody.
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = '60s'")
        for table, column in _CHECKS:
            op.execute(
                f"ALTER TABLE benchmarks_v2.{table} VALIDATE CONSTRAINT {table}_{column}_check"
            )
        op.execute("RESET lock_timeout")


def upgrade() -> None:
    """Admit 'LLM' everywhere a benchmark is stored, then seed the Phonely row."""
    _swap_checks(_WIDE, validate_inline=False)
    # Idempotent: the validate step commits before the version is stamped, so a
    # retry after a failure there must not duplicate the row.
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
    _validate_checks()


def downgrade() -> None:
    """Remove the seed row unless edited since, then re-narrow (fails if LLM rows remain)."""
    op.get_bind().exec_driver_sql(
        """
        DELETE FROM benchmarks_v2.models
        WHERE updated_by_user_id = 'migration:20260901_0025'
          AND modality = 'LLM' AND provider = 'phonely' AND model = 'phonely-agent';
        """
    )
    # Validated inline: a narrowing that would strand LLM rows must fail atomically.
    _swap_checks(_NARROW, validate_inline=True)
