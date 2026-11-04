# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Seed the CharacterErrorRate and ToolCallCount metrics; rows are trigger-retained."""

from __future__ import annotations

from alembic import op

revision = "20261104_0048"
down_revision = "20261102_0047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """INSERT INTO benchmarks_v2.metrics (code, display_name)
           VALUES ('CharacterErrorRate', 'Character Error Rate'),
                  ('ToolCallCount', 'Tool Call Count')
           ON CONFLICT (code) DO NOTHING"""
    )


def downgrade() -> None:
    pass
