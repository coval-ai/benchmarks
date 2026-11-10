# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Seed the interruption suites' metrics; rows are trigger-retained."""

from __future__ import annotations

from alembic import op

revision = "20261107_0050"
down_revision = "20261107_0049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """INSERT INTO benchmarks_v2.metrics (code, display_name)
           VALUES ('InterruptionCueLanded', 'Interruption Cue Landed'),
                  ('InterruptionYielded', 'Interruption Yielded'),
                  ('BackchannelContinued', 'Backchannel Continued')
           ON CONFLICT (code) DO NOTHING"""
    )


def downgrade() -> None:
    pass
