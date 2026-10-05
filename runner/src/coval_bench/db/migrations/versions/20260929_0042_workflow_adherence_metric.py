# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Seed the WorkflowAdherence metric; the row is trigger-retained, so downgrade is a no-op."""

from __future__ import annotations

from alembic import op

revision = "20260929_0042"
down_revision = "20260921_0041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """INSERT INTO benchmarks_v2.metrics (code, display_name)
           VALUES ('WorkflowAdherence', 'Workflow Adherence')
           ON CONFLICT (code) DO NOTHING"""
    )


def downgrade() -> None:
    pass
