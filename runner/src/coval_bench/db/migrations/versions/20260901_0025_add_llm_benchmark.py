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
constraint, so they need no change (same shape as 20260707_0009).

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

_CHECKS: tuple[tuple[str, str], ...] = (
    ("results", "benchmark"),
    ("results_by_bucket", "benchmark"),
    ("benchmark_observations", "benchmark"),
    ("metric_values_by_bucket", "benchmark"),
    ("models", "modality"),
    ("model_history", "modality"),
)


def _set_checks(allowed: str) -> None:
    for table, column in _CHECKS:
        name = f"{table}_{column}_check"
        op.execute(f"ALTER TABLE benchmarks_v2.{table} DROP CONSTRAINT IF EXISTS {name}")  # noqa: S608
        op.execute(
            f"ALTER TABLE benchmarks_v2.{table} ADD CONSTRAINT {name} CHECK ({column} IN ({allowed}))"  # noqa: S608
        )


def upgrade() -> None:
    """Admit 'LLM' everywhere a benchmark is stored, then seed the Phonely row."""
    _set_checks("'STT','TTS','S2S','LLM'")
    op.get_bind().exec_driver_sql(
        """
        INSERT INTO benchmarks_v2.models
            (modality, provider, model, voice, voices, creator, source, licensing,
             on_prem, region, arena_enabled, collected, published, updated_by_user_id)
        VALUES
            ('LLM', 'phonely', 'phonely-agent', NULL, '[]'::jsonb, NULL, 'official-api', 'proprietary', FALSE, 'us', FALSE, TRUE, FALSE, 'migration:20260901_0025');
        """
    )


def downgrade() -> None:
    """Remove the seed row unless edited since, then re-narrow (fails if LLM rows remain)."""
    op.get_bind().exec_driver_sql(
        """
        DELETE FROM benchmarks_v2.models
        WHERE updated_by_user_id = 'migration:20260901_0025'
          AND modality = 'LLM' AND provider = 'phonely' AND model = 'phonely-agent';
        """
    )
    _set_checks("'STT','TTS','S2S'")
