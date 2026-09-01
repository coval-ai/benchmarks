# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Register the private Phonely text-agent benchmark entry.

Revision ID: 20260901_0025
Revises:     20260831_0024
Create Date: 2026-09-01
"""

from __future__ import annotations

from alembic import op

revision = "20260901_0025"
down_revision = "20260831_0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Insert the collected, unpublished Phonely entry."""
    op.get_bind().exec_driver_sql(
        """
        INSERT INTO benchmarks_v2.models
            (modality, provider, model, voice, voices, creator, source, licensing,
             on_prem, region, arena_enabled, collected, published, updated_by_user_id)
        VALUES
            ('S2S', 'phonely', 'phonely-agent', NULL, '[]'::jsonb, NULL,
             'official-api', 'proprietary', FALSE, 'us', FALSE, TRUE, FALSE,
             'migration:20260901_0025');
        """
    )


def downgrade() -> None:
    """Remove the Phonely entry when it has not been edited since migration."""
    op.get_bind().exec_driver_sql(
        """
        DELETE FROM benchmarks_v2.models
        WHERE updated_by_user_id = 'migration:20260901_0025'
          AND modality = 'S2S'
          AND provider = 'phonely'
          AND model = 'phonely-agent';
        """
    )
