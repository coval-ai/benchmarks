# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Arena battles: fold pre-domain battles into ``other``.

Battles created before the domain dropdown have ``domain`` NULL (confirmed the only
value in prod as of 2026-07-15); the API now only accepts the fixed domain set, so
tag them ``other``. Their votes already count toward the aggregate ``all`` board,
which ignores domain — this only adds them to the ``other`` per-domain board.

Irreversible: which rows were NULL is not preserved, so ``downgrade`` is a no-op.
"""

from __future__ import annotations

from alembic import op

revision = "20260715_0011"
down_revision = "20260715_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE arena.battles SET domain = 'other' WHERE domain IS NULL")


def downgrade() -> None:
    pass
