# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality sweeps.

Revision ID: 20261112_0055
Revises:     20261112_0054
Create Date: 2026-11-12

A sweep runs every item of one dataset through one model version. ``sweeps`` is
one row per model version and dataset version.
"""

from __future__ import annotations

from alembic import op

revision = "20261112_0055"
down_revision = "20261112_0054"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE benchmarks_v2.sweeps (
            id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            model_id        BIGINT NOT NULL REFERENCES benchmarks_v2.models (id),
            model_version   INTEGER NOT NULL CHECK (model_version >= 1),
            dataset_id      TEXT NOT NULL CHECK (dataset_id <> ''),
            dataset_version TEXT NOT NULL CHECK (dataset_version <> ''),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (model_id, model_version, dataset_id, dataset_version)
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS benchmarks_v2.sweeps;")
