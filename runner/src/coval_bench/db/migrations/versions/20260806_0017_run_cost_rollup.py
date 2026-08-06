# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-run spend rollup columns.

``total_cost_usd`` = the run's COST_USD result rows + the whisper judge's own
spend (``judge_cost_usd``), computed at ``finish_run``. Nullable, no backfill:
runs before cost capture stay NULL rather than reporting a fabricated zero.
Plain ADD COLUMN on ``runs`` — no matview involvement.
"""

from __future__ import annotations

from alembic import op

revision = "20260806_0017"
down_revision = "20260806_0016"
branch_labels = None
depends_on = None

_COLUMNS: tuple[str, ...] = ("total_cost_usd", "judge_cost_usd")


def upgrade() -> None:
    for column in _COLUMNS:
        op.execute(f"ALTER TABLE benchmarks_v2.runs ADD COLUMN {column} NUMERIC(12,4)")


def downgrade() -> None:
    for column in _COLUMNS:
        op.execute(f"ALTER TABLE benchmarks_v2.runs DROP COLUMN {column}")
