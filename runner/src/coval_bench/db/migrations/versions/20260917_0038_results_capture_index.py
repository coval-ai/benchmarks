# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the capture-time index used by the v2 results cursor query."""

from __future__ import annotations

from alembic import op

revision = "20260917_0038"
down_revision = "20260916_0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Replace any interrupted or older same-name index, then build the index."""
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "benchmarks_v2.benchmark_observations_capture_order_idx"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY benchmark_observations_capture_order_idx "
            "ON benchmarks_v2.benchmark_observations (captured_at DESC)"
        )


def downgrade() -> None:
    """Remove the capture-order index without blocking observation writes."""
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "benchmarks_v2.benchmark_observations_capture_order_idx"
        )
