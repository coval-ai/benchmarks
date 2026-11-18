# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Give each benchmark a visibility level; new benchmarks start coval-only.

Revision ID: 20261117_0068
Revises:     20261117_0067
Create Date: 2026-11-17

The same levels a model and a metric have. Every benchmark that exists stays public.
"""

from __future__ import annotations

from alembic import op

revision = "20261117_0068"
down_revision = "20261117_0067"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the level, public for the benchmarks that exist and coval for new ones."""
    op.execute(
        """
        ALTER TABLE benchmarks_v2.benchmarks
            ADD COLUMN visibility TEXT NOT NULL DEFAULT 'public'
                CHECK (visibility IN ('public', 'partner', 'coval'));
        ALTER TABLE benchmarks_v2.benchmarks ALTER COLUMN visibility SET DEFAULT 'coval';
        """
    )


def downgrade() -> None:
    """Drop the level."""
    op.execute("ALTER TABLE benchmarks_v2.benchmarks DROP COLUMN visibility")
