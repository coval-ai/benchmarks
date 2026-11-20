# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Give each metric a collected flag; the ones nothing runs start off."""

from __future__ import annotations

from alembic import op

revision = "20261120_0070"
down_revision = "20261119_0069"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '10s'")
    op.execute(
        "ALTER TABLE benchmarks_v2.metrics ADD COLUMN collected BOOLEAN NOT NULL DEFAULT TRUE"
    )
    op.execute(
        """UPDATE benchmarks_v2.metrics
           SET collected = FALSE
           WHERE code IN ('SemDistMiniLM', 'JudgeJev', 'ExpectedBehaviorAdherence')"""
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '10s'")
    op.execute("ALTER TABLE benchmarks_v2.metrics DROP COLUMN collected")
