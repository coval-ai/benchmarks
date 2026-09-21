# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the normalized importer dedup lookup index."""

from __future__ import annotations

from alembic import op

revision = "20260921_0041"
down_revision = "20260918_0040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Replace an interrupted or stale same-name index, then build it."""
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "benchmarks_v2.benchmark_observations_coval_ingest_idx"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY benchmark_observations_coval_ingest_idx "
            "ON benchmarks_v2.benchmark_observations "
            "(provider, benchmark, (split_part(sample_id, '/', 1))) "
            "INCLUDE (id, run_id)"
        )


def downgrade() -> None:
    """Remove the normalized importer lookup index without blocking writes."""
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "benchmarks_v2.benchmark_observations_coval_ingest_idx"
        )
