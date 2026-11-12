# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep results.

Revision ID: 20261112_0056
Revises:     20261112_0055
Create Date: 2026-11-12

``sweep_results`` is one row per dataset item in a sweep, with the transcript
and the raw timings the provider call produced, or the error if it failed. A
sweep with a result for every item gets ``completed_at``.
"""

from __future__ import annotations

from alembic import op

revision = "20261112_0056"
down_revision = "20261112_0055"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.sweeps ADD COLUMN completed_at TIMESTAMPTZ;

        CREATE TABLE benchmarks_v2.sweep_results (
            id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            sweep_id   BIGINT NOT NULL REFERENCES benchmarks_v2.sweeps (id),
            sample_id  TEXT NOT NULL CHECK (sample_id <> ''),
            transcript TEXT,
            timings    JSONB NOT NULL DEFAULT '{}'::jsonb,
            error      TEXT CHECK (error <> ''),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (sweep_id, sample_id)
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS benchmarks_v2.sweep_results;
        ALTER TABLE benchmarks_v2.sweeps DROP COLUMN completed_at;
        """
    )
