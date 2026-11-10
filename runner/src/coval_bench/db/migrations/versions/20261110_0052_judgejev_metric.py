# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Seed the JudgeJev metric; the row is trigger-retained."""

from __future__ import annotations

from alembic import op

revision = "20261110_0052"
down_revision = "20261109_0051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """INSERT INTO benchmarks_v2.metrics (code, display_name)
           VALUES ('JudgeJev', 'JudgeJev')
           ON CONFLICT (code) DO NOTHING"""
    )


def downgrade() -> None:
    pass
