# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Append-only ``model_pricing`` ratesheet.

Rates live in the DB, not the code registry, because they change on their own
schedule and history matters: a June run is costed at June rates. A rate is
never UPDATEd — a new row is inserted and ``superseded_at`` is stamped on the
old one, so the partial unique index keeps exactly one effective row per
(provider, model, benchmark, billing_unit). Token-billed models legitimately
hold two effective rows (input + output units). ``benchmark`` is part of the
key because (provider, model) alone is not unique in the registry — gradium
serves STT and TTS under the same model id. Rates are stored in the provider's
native billing unit; normalization happens at read time.
"""

from __future__ import annotations

from alembic import op

revision = "20260806_0016"
down_revision = "20260806_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE benchmarks_v2.model_pricing (
          id BIGSERIAL PRIMARY KEY,
          provider TEXT NOT NULL,
          model TEXT NOT NULL,
          benchmark TEXT NOT NULL,
          billing_unit TEXT NOT NULL,
          rate_usd NUMERIC(14,8) NOT NULL,
          plan_assumption TEXT NULL,
          effective_at TIMESTAMPTZ NOT NULL,
          superseded_at TIMESTAMPTZ NULL,
          source_url TEXT NOT NULL,
          as_of DATE NOT NULL,
          evidence TEXT NULL,
          updated_by TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX ux_model_pricing_effective
          ON benchmarks_v2.model_pricing (provider, model, benchmark, billing_unit)
          WHERE superseded_at IS NULL
        """
    )
    op.execute(
        """
        CREATE INDEX ix_model_pricing_lookup
          ON benchmarks_v2.model_pricing (provider, model, effective_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE benchmarks_v2.model_pricing")
