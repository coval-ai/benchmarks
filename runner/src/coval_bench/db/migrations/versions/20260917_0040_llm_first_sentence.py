# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Add time to first sentence to the proxy's per-turn LLM timing."""

from __future__ import annotations

from alembic import op

revision = "20260917_0040"
down_revision = "20260917_0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.llm_turns
            ADD COLUMN first_sentence_ms DOUBLE PRECISION CHECK (
                first_sentence_ms IS NULL
                OR (first_sentence_ms >= ttft_ms AND first_sentence_ms <= total_ms)
            ),
            ADD COLUMN first_sentence_chars INTEGER CHECK (
                first_sentence_chars IS NULL OR first_sentence_chars > 0
            )
        """
    )
    op.execute(
        """
        INSERT INTO benchmarks_v2.metrics (code, display_name)
        VALUES ('TimeToFirstSentence', 'Time to First Sentence')
        ON CONFLICT (code) DO NOTHING
        """
    )


def downgrade() -> None:
    # Metric definitions are retained by trigger, so only the columns come back out.
    op.execute(
        """
        ALTER TABLE benchmarks_v2.llm_turns
            DROP COLUMN first_sentence_ms,
            DROP COLUMN first_sentence_chars
        """
    )
