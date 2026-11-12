# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Model version: models.version.

Revision ID: 20261112_0054
Revises:     20261111_0053
Create Date: 2026-11-12

Our own count of how many times a model has been measured from scratch. It isn't
the provider's version. Every model starts at 1.
"""

from __future__ import annotations

from alembic import op

revision = "20261112_0054"
down_revision = "20261111_0053"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.models
            ADD COLUMN version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1);
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE benchmarks_v2.models DROP COLUMN version;")
