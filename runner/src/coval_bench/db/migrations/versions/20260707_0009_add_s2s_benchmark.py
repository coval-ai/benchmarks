# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Allow 'S2S' in the benchmark CHECK on results and results_by_bucket.

Both tables were created with an inline, column-level CHECK
(``benchmark IN ('STT','TTS')``), which Postgres names deterministically
``<table>_benchmark_check``. Adding the S2S dashboard requires widening both
to include ``'S2S'``. The per-window matviews (results_24h/7d/30d) derive from
``results`` and carry no constraint, so they need no change.
"""

from __future__ import annotations

from alembic import op

revision = "20260707_0009"
down_revision = "20260626_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Widen both benchmark CHECKs to include 'S2S'."""
    op.execute(
        "ALTER TABLE benchmarks_v2.results DROP CONSTRAINT IF EXISTS results_benchmark_check"
    )
    op.execute(
        "ALTER TABLE benchmarks_v2.results "
        "ADD CONSTRAINT results_benchmark_check "
        "CHECK (benchmark IN ('STT','TTS','S2S'))"
    )
    op.execute(
        "ALTER TABLE benchmarks_v2.results_by_bucket "
        "DROP CONSTRAINT IF EXISTS results_by_bucket_benchmark_check"
    )
    op.execute(
        "ALTER TABLE benchmarks_v2.results_by_bucket "
        "ADD CONSTRAINT results_by_bucket_benchmark_check "
        "CHECK (benchmark IN ('STT','TTS','S2S'))"
    )


def downgrade() -> None:
    """Restore the STT/TTS-only CHECKs (fails if any S2S rows exist)."""
    op.execute(
        "ALTER TABLE benchmarks_v2.results DROP CONSTRAINT IF EXISTS results_benchmark_check"
    )
    op.execute(
        "ALTER TABLE benchmarks_v2.results "
        "ADD CONSTRAINT results_benchmark_check "
        "CHECK (benchmark IN ('STT','TTS'))"
    )
    op.execute(
        "ALTER TABLE benchmarks_v2.results_by_bucket "
        "DROP CONSTRAINT IF EXISTS results_by_bucket_benchmark_check"
    )
    op.execute(
        "ALTER TABLE benchmarks_v2.results_by_bucket "
        "ADD CONSTRAINT results_by_bucket_benchmark_check "
        "CHECK (benchmark IN ('STT','TTS'))"
    )
