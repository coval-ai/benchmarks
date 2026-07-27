# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Enable pg_stat_statements for query-level performance stats.

Creating the extension only registers the view and functions; collection
requires the library in ``shared_preload_libraries``, which Cloud SQL
preloads by default. Local Postgres does not, so the view exists there but
errors when queried.
"""

from __future__ import annotations

from alembic import op

revision = "20260727_0011"
down_revision = "20260715_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the pg_stat_statements extension."""
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements SCHEMA public")


def downgrade() -> None:
    """Drop the pg_stat_statements extension."""
    op.execute("DROP EXTENSION IF EXISTS pg_stat_statements")
