# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Add normalized dashboard-read indexes.

CONCURRENTLY avoids blocking runner writes during the builds, so the statements
must run outside a transaction. An interrupted build can leave INVALID indexes;
drop the affected cleanup target manually before re-running the migration:

    DROP INDEX benchmarks_v2.benchmark_observations_recent_results_idx;
    DROP INDEX benchmarks_v2.metric_values_by_bucket_series_idx;
"""

from __future__ import annotations

from alembic import op

revision = "20260824_0019"
down_revision = "20260818_0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add read indexes without blocking normalized-storage writers."""
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY benchmark_observations_recent_results_idx "
            "ON benchmarks_v2.benchmark_observations "
            "(benchmark, dataset_id, captured_at DESC, id DESC) "
            "WHERE status = 'succeeded'"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY metric_values_by_bucket_series_idx "
            "ON benchmarks_v2.metric_values_by_bucket "
            "(benchmark, dataset_id, metric_version, evaluation_variant, value_key, bucket_at)"
        )


def downgrade() -> None:
    """Remove the normalized dashboard-read indexes without blocking writers."""
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "benchmarks_v2.benchmark_observations_recent_results_idx"
        )
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS benchmarks_v2.metric_values_by_bucket_series_idx"
        )
