# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw pricing-page snapshots backing the price collector's evidence trail.

Every ``model_pricing`` row a bot writes references a snapshot hash; keeping
the gzipped page text lets a reviewer see exactly what the extractor saw. A
new row is stored only when the page hash changes, so weekly runs against
static pages cost nothing.
"""

from __future__ import annotations

from alembic import op

revision = "20260806_0018"
down_revision = "20260806_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE benchmarks_v2.pricing_snapshots (
          id BIGSERIAL PRIMARY KEY,
          provider TEXT NOT NULL,
          url TEXT NOT NULL,
          sha256 TEXT NOT NULL,
          content_gz BYTEA NOT NULL,
          fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX ix_pricing_snapshots_provider
          ON benchmarks_v2.pricing_snapshots (provider, fetched_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE benchmarks_v2.pricing_snapshots")
