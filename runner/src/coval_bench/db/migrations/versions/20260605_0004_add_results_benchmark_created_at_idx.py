# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Add (benchmark, created_at) index on results.

The TTS/STT dashboards (GET /v1/results) and the live 7d/30d leaderboard
filter results by benchmark plus a created_at window. No existing index
covers that path — results_provider_model_idx leads with provider/model,
which those queries don't filter on — so they seq-scan the whole table.

CONCURRENTLY avoids blocking runner inserts during the build, so it must
run outside a transaction (autocommit_block). If the build fails it leaves
an INVALID index; drop it manually before re-running migrate:

    DROP INDEX benchmarks_v2.results_benchmark_created_at_idx;
"""

from __future__ import annotations

from alembic import op

revision = "20260605_0004"
down_revision = "20260602_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the (benchmark, created_at DESC) index without blocking writes."""
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY results_benchmark_created_at_idx "
            "ON benchmarks_v2.results (benchmark, created_at DESC)"
        )


def downgrade() -> None:
    """Drop the index."""
    op.execute("DROP INDEX IF EXISTS benchmarks_v2.results_benchmark_created_at_idx")
