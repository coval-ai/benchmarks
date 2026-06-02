# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Add scheduled_at to runs and backfill from started_at.

scheduled_at is the cron trigger time, snapshotted once at run start and
floored to the scheduler period so every result in a run shares one bucket.
Historical rows are backfilled with floor(started_at / 1800) — the 30-minute
grid the chart used to compute client-side — so existing points don't shift.
"""

from __future__ import annotations

from alembic import op

revision = "20260602_0003"
down_revision = "20260601_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add nullable scheduled_at column and backfill existing rows."""
    op.execute("SET search_path TO benchmarks_v2")
    op.execute("ALTER TABLE runs ADD COLUMN scheduled_at TIMESTAMPTZ")
    op.execute(
        """
        UPDATE runs
        SET scheduled_at =
            to_timestamp(floor(extract(epoch FROM started_at) / 1800) * 1800)
        WHERE scheduled_at IS NULL
        """
    )


def downgrade() -> None:
    """Drop the scheduled_at column."""
    op.execute("SET search_path TO benchmarks_v2")
    op.execute("ALTER TABLE runs DROP COLUMN scheduled_at")
