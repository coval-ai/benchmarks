# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Arena votes: dedup per identity, not per (identity, voter_type).

Replace ``UNIQUE (battle_id, voter_type, voter_id)`` with ``UNIQUE (battle_id, voter_id)`` so
one identity has at most one current vote per battle regardless of voter_type (a re-vote
stays an UPDATE via ``ON CONFLICT``).
"""

from __future__ import annotations

from alembic import op

revision = "20260626_0008"
down_revision = "20260615_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE arena.votes DROP CONSTRAINT votes_battle_id_voter_type_voter_id_key")
    op.execute(
        "ALTER TABLE arena.votes "
        "ADD CONSTRAINT votes_battle_id_voter_id_key UNIQUE (battle_id, voter_id)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE arena.votes DROP CONSTRAINT votes_battle_id_voter_id_key")
    op.execute(
        "ALTER TABLE arena.votes "
        "ADD CONSTRAINT votes_battle_id_voter_type_voter_id_key "
        "UNIQUE (battle_id, voter_type, voter_id)"
    )
