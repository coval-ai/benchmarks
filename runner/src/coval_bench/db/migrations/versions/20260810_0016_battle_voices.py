# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Record which voice sang each side of an arena battle, and the gender they share.

Gender is stored on the row rather than derived from the voice ids at read time:
voices get retired from the registry as providers change their catalogues, and a
retired id would leave historical battles unattributable. Voice ids are kept
alongside it for auditing which speaker a rating was earned with.

Nullable with no backfill. Battles predating this migration were cross-gender by
construction, so a NULL gender is exactly the marker the ``davidson-bt-002``
rating methodology uses to exclude them — no date cutoff needed.
"""

from __future__ import annotations

from alembic import op

revision = "20260810_0016"
down_revision = "20260807_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE arena.battles ADD COLUMN voice_a TEXT")
    op.execute("ALTER TABLE arena.battles ADD COLUMN voice_b TEXT")
    op.execute(
        "ALTER TABLE arena.battles ADD COLUMN gender TEXT CHECK (gender IN ('female', 'male'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE arena.battles DROP COLUMN gender")
    op.execute("ALTER TABLE arena.battles DROP COLUMN voice_b")
    op.execute("ALTER TABLE arena.battles DROP COLUMN voice_a")
