# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Give each model a price.

Revision ID: 20261124_0072
Revises:     20261120_0071
Create Date: 2026-11-24

One price per model, in US dollars, null when it has none. The unit follows the
modality: per hour of audio for STT and per 1M characters for TTS.
"""

from __future__ import annotations

from alembic import op

revision = "20261124_0072"
down_revision = "20261120_0071"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.models
            ADD COLUMN price_usd NUMERIC CHECK (price_usd >= 0);
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE benchmarks_v2.models DROP COLUMN price_usd;")
