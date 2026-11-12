# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep scores.

Revision ID: 20261112_0057
Revises:     20261112_0056
Create Date: 2026-11-12

``sweep_scores`` is one row per sweep result, metric and metric version. It
holds the error and reference counts, so a rate is their ratio and can be pooled
across a whole sweep.
"""

from __future__ import annotations

from alembic import op

revision = "20261112_0057"
down_revision = "20261112_0056"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE benchmarks_v2.sweep_scores (
            id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            sweep_result_id BIGINT NOT NULL REFERENCES benchmarks_v2.sweep_results (id),
            metric          TEXT NOT NULL CHECK (metric <> ''),
            metric_version  TEXT NOT NULL CHECK (metric_version <> ''),
            errors          INTEGER NOT NULL CHECK (errors >= 0),
            reference_units INTEGER NOT NULL CHECK (reference_units >= 0),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (sweep_result_id, metric, metric_version)
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.sweep_scores;")
