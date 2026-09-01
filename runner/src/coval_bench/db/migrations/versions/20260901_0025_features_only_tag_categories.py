# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Narrow the tag category check to the one category that exists.

Revision ID: 20260901_0025
Revises:     20260831_0024
Create Date: 2026-09-01

Nothing serves a second curated category: ``TagRecord`` pins the column to
``features`` and no row has held anything else.
"""

from __future__ import annotations

from alembic import op

revision = "20260901_0025"
down_revision = "20260831_0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Drop the two-value check for a features-only one."""
    op.get_bind().exec_driver_sql(
        """
        ALTER TABLE benchmarks_v2.tags DROP CONSTRAINT tags_category_check;
        ALTER TABLE benchmarks_v2.tags
            ADD CONSTRAINT tags_category_check CHECK (category = 'features');
        """
    )


def downgrade() -> None:
    """Restore the check that also allowed 'mode'."""
    op.get_bind().exec_driver_sql(
        """
        ALTER TABLE benchmarks_v2.tags DROP CONSTRAINT tags_category_check;
        ALTER TABLE benchmarks_v2.tags
            ADD CONSTRAINT tags_category_check CHECK (category IN ('mode', 'features'));
        """
    )
