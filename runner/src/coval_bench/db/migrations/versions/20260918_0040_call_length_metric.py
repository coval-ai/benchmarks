# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Seed the CallLength metric identity; the row is trigger-retained, so downgrade is a no-op."""

from __future__ import annotations

from alembic import op

revision = "20260918_0040"
down_revision = "20260917_0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """INSERT INTO benchmarks_v2.metrics (code, display_name)
           VALUES ('CallLength', 'Call Length')
           ON CONFLICT (code) DO NOTHING"""
    )


def downgrade() -> None:
    pass
