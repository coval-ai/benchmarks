# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe results model id.

Revision ID: 20261112_0062
Revises:     20261112_0061
Create Date: 2026-11-12

``probe_results`` points at ``models.id`` like the sweep tables do, so a renamed
model keeps its probes. The modality, provider and model name are read from the
model row. A probe of a model that has no row is dropped.
"""

from __future__ import annotations

from alembic import op

revision = "20261112_0062"
down_revision = "20261112_0061"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.probe_results
            ADD COLUMN model_id BIGINT REFERENCES benchmarks_v2.models (id);
        UPDATE benchmarks_v2.probe_results p SET model_id = m.id
            FROM benchmarks_v2.models m
            WHERE m.modality = p.benchmark AND m.provider = p.provider AND m.model = p.model;
        DELETE FROM benchmarks_v2.probe_results WHERE model_id IS NULL;
        ALTER TABLE benchmarks_v2.probe_results
            ALTER COLUMN model_id SET NOT NULL,
            DROP COLUMN benchmark,
            DROP COLUMN provider,
            DROP COLUMN model,
            ADD UNIQUE (model_id, scheduled_at);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.probe_results
            ADD COLUMN benchmark TEXT CHECK (benchmark IN ('STT', 'TTS')),
            ADD COLUMN provider TEXT CHECK (provider <> ''),
            ADD COLUMN model TEXT CHECK (model <> '');
        UPDATE benchmarks_v2.probe_results p
            SET benchmark = m.modality, provider = m.provider, model = m.model
            FROM benchmarks_v2.models m
            WHERE m.id = p.model_id;
        ALTER TABLE benchmarks_v2.probe_results
            ALTER COLUMN benchmark SET NOT NULL,
            ALTER COLUMN provider SET NOT NULL,
            ALTER COLUMN model SET NOT NULL,
            DROP COLUMN model_id,
            ADD UNIQUE (benchmark, provider, model, scheduled_at);
        """
    )
