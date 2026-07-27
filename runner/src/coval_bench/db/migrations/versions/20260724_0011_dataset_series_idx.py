# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cover dataset_id in the results_by_bucket series index.

The aggregates series query filters on (benchmark, dataset_id, bucket_at);
the previous (benchmark, bucket_at) index made every per-dataset read scan
all datasets' rows for the window.
"""

from __future__ import annotations

from alembic import op

revision = "20260724_0011"
down_revision = "20260715_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Replace the series index with one that covers dataset_id."""
    op.execute("DROP INDEX IF EXISTS benchmarks_v2.results_by_bucket_series_idx")
    op.execute(
        "CREATE INDEX results_by_bucket_series_idx "
        "ON benchmarks_v2.results_by_bucket (benchmark, dataset_id, bucket_at)"
    )


def downgrade() -> None:
    """Restore the pre-dataset series index."""
    op.execute("DROP INDEX IF EXISTS benchmarks_v2.results_by_bucket_series_idx")
    op.execute(
        "CREATE INDEX results_by_bucket_series_idx "
        "ON benchmarks_v2.results_by_bucket (benchmark, bucket_at)"
    )
