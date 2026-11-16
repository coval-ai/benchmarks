# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Give each metric a visibility level; new metrics start coval-only."""

from __future__ import annotations

from alembic import op

revision = "20261116_0064"
down_revision = "20261112_0063"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '10s'")
    op.execute(
        """ALTER TABLE benchmarks_v2.metrics
             ADD COLUMN visibility TEXT NOT NULL DEFAULT 'public'
               CHECK (visibility IN ('public', 'partner', 'coval'))"""
    )
    op.execute(
        """UPDATE benchmarks_v2.metrics
           SET visibility = 'coval'
           WHERE code IN ('InterruptionCueLanded', 'InterruptionYielded', 'BackchannelContinued',
                          'InterruptionRate', 'CharacterErrorRate', 'ToolCallCount')"""
    )
    op.execute("ALTER TABLE benchmarks_v2.metrics ALTER COLUMN visibility SET DEFAULT 'coval'")


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '10s'")
    op.execute("ALTER TABLE benchmarks_v2.metrics DROP COLUMN visibility")
