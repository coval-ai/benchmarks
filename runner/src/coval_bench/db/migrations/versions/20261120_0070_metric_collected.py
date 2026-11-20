# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Give each metric a collected flag; every metric starts collected."""

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


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '10s'")
    op.execute("ALTER TABLE benchmarks_v2.metrics DROP COLUMN collected")
