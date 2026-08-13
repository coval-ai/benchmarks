# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Record the Coval caller persona on the run row.

Provenance only: nothing groups, filters or aggregates by it, so the stats
matviews and results_by_bucket are untouched. The caller condition the dashboard
slices on rides on ``dataset_id`` instead, because the two clean personas are
deliberately pooled and only the noisy one is a separate condition. Storing the
id anyway means per-persona history exists if that slicing is ever wanted.

Nullable, no backfill: pre-migration runs keep NULL, and non-S2S runs have no
persona at all.
"""

from __future__ import annotations

from alembic import op

revision = "20260812_0017"
down_revision = "20260810_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the nullable persona_id column to runs."""
    op.execute("ALTER TABLE benchmarks_v2.runs ADD COLUMN persona_id TEXT")


def downgrade() -> None:
    """Drop the persona_id column."""
    op.execute("ALTER TABLE benchmarks_v2.runs DROP COLUMN persona_id")
