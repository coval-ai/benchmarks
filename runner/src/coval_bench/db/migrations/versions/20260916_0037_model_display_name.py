# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-model display names: models.display_name.

Revision ID: 20260916_0037
Revises:     20260915_0036
Create Date: 2026-09-16

The site labels each model from a slug-to-name map compiled into the frontend, so
naming a new model means a deploy. The name now lives on the model row, edited from
the admin registry like any other field, and ``/v1/providers`` carries it.

NULL means the site still labels the model itself: nothing is seeded, and the
frontend keeps its compiled map as the fallback until every row is filled in.
"""

from __future__ import annotations

from alembic import op

revision = "20260916_0037"
down_revision = "20260915_0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.models
            ADD COLUMN display_name TEXT CHECK (display_name IS NULL OR display_name <> '');
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE benchmarks_v2.models DROP COLUMN display_name;")
