# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Embedded SQL is formatted as executable migration text.

"""Pricing rate change-log: pricing_rates.

Revision ID: 20260826_0022
Revises:     20260825_0021
Create Date: 2026-08-26

Append-only record of every published rate the pricing registry has carried,
so cost is trackable over time rather than a snapshot. ``coval-bench pricing
sync`` records the packaged ratesheets into it and is idempotent: the unique
key makes re-recording an unchanged registry a no-op, while a new price or a
new effective date appends. Notes/source edits alone do not re-record — this
tracks prices, not prose.

No FK to ``models``: like ``model_history``, history must outlive anything it
references. No seed migration: the first sync run is the seed, and stays the
ongoing mechanism.

Notes
-----
DB roles (``runner``, ``api``) and GRANTs are managed by Terraform. The sync
runs inside the runner job, so the ``runner`` role needs SELECT and INSERT.
Nothing is ever updated or deleted.
"""

from __future__ import annotations

from alembic import op

revision = "20260826_0022"
down_revision = "20260825_0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the pricing rate change-log."""
    op.execute(
        """
        CREATE TABLE benchmarks_v2.pricing_rates (
            id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            benchmark      TEXT NOT NULL CHECK (benchmark IN ('STT', 'TTS', 'S2S')),
            provider       TEXT NOT NULL CHECK (provider <> ''),
            model          TEXT NOT NULL CHECK (model <> ''),
            unit           TEXT NOT NULL CHECK (unit <> ''),
            -- NUMERIC keeps the registry's printed scale: 0.20 stays 0.20,
            -- never 0.2 — the same exact-decimal contract /v1/pricing serves.
            price_usd      NUMERIC NOT NULL CHECK (price_usd > 0),
            effective_from DATE NOT NULL,
            source_url     TEXT NOT NULL CHECK (source_url <> ''),
            notes          TEXT CHECK (notes IS NULL OR notes <> ''),
            recorded_by    TEXT NOT NULL CHECK (recorded_by <> ''),
            recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            -- One row per distinct rate. A price revert carries a new
            -- effective_from (the day it took effect again), so round trips
            -- still append; only a byte-identical rate is skipped.
            UNIQUE (benchmark, provider, model, unit, price_usd, effective_from)
        );
        CREATE INDEX pricing_rates_key_recorded_at
            ON benchmarks_v2.pricing_rates (benchmark, provider, model, recorded_at DESC);
        """
    )


def downgrade() -> None:
    """Drop the pricing rate change-log."""
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.pricing_rates;")
