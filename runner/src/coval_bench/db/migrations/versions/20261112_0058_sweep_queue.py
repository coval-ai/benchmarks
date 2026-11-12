# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep queue.

Revision ID: 20261112_0058
Revises:     20261112_0057
Create Date: 2026-11-12

``sweep_queue`` is one row per dataset item a sweep still has to run, with the
tries it has left. A row is deleted when its item gets a result, and stays at
zero tries when the provider failed it every time. ``sweep_results`` stops
holding failures, so its ``error`` column goes.
"""

from __future__ import annotations

from alembic import op

revision = "20261112_0058"
down_revision = "20261112_0057"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE benchmarks_v2.sweep_queue (
            sweep_id   BIGINT NOT NULL REFERENCES benchmarks_v2.sweeps (id),
            sample_id  TEXT NOT NULL CHECK (sample_id <> ''),
            tries_left INTEGER NOT NULL DEFAULT 3 CHECK (tries_left >= 0),
            last_error TEXT CHECK (last_error <> ''),
            PRIMARY KEY (sweep_id, sample_id)
        );

        ALTER TABLE benchmarks_v2.sweep_results DROP COLUMN error;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.sweep_results ADD COLUMN error TEXT CHECK (error <> '');
        DROP TABLE IF EXISTS benchmarks_v2.sweep_queue;
        """
    )
