# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Give each model a visibility level; ``published`` is worked out from it.

Revision ID: 20261117_0067
Revises:     20261117_0066
Create Date: 2026-11-17

A model is public, partner or coval, the same levels a metric has. A published model
becomes public and an unpublished one partner, so no one sees more or less than before.

``published`` stays as a generated column, true for a public model, so everything that
reads it keeps working. Nothing can write it any more.
"""

from __future__ import annotations

from alembic import op

revision = "20261117_0067"
down_revision = "20261117_0066"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the level, fill it from ``published``, and make ``published`` follow it."""
    op.execute(
        """
        ALTER TABLE benchmarks_v2.models
            ADD COLUMN visibility TEXT NOT NULL DEFAULT 'partner'
                CHECK (visibility IN ('public', 'partner', 'coval'));
        UPDATE benchmarks_v2.models SET visibility = 'public' WHERE published;
        ALTER TABLE benchmarks_v2.models DROP COLUMN published;
        ALTER TABLE benchmarks_v2.models
            ADD COLUMN published BOOLEAN GENERATED ALWAYS AS (visibility = 'public') STORED;
        """
    )


def downgrade() -> None:
    """Put ``published`` back as a plain column. A coval model becomes unpublished."""
    op.execute(
        """
        ALTER TABLE benchmarks_v2.models DROP COLUMN published;
        ALTER TABLE benchmarks_v2.models ADD COLUMN published BOOLEAN NOT NULL DEFAULT false;
        UPDATE benchmarks_v2.models SET published = true WHERE visibility = 'public';
        ALTER TABLE benchmarks_v2.models DROP COLUMN visibility;
        """
    )
