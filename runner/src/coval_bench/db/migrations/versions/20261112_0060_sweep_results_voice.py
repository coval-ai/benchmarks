# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep results voice.

Revision ID: 20261112_0060
Revises:     20261112_0059
Create Date: 2026-11-12

A TTS result is for a prompt in a voice, so ``sweep_results`` gets the voice the
queue row had. STT rows keep an empty voice.
"""

from __future__ import annotations

from alembic import op

revision = "20261112_0060"
down_revision = "20261112_0059"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.sweep_results
            ADD COLUMN voice TEXT NOT NULL DEFAULT '',
            DROP CONSTRAINT sweep_results_sweep_id_sample_id_key,
            ADD UNIQUE (sweep_id, sample_id, voice);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DELETE FROM benchmarks_v2.sweep_scores s USING benchmarks_v2.sweep_results r
            WHERE r.id = s.sweep_result_id AND r.voice <> '';
        DELETE FROM benchmarks_v2.sweep_results WHERE voice <> '';
        ALTER TABLE benchmarks_v2.sweep_results
            DROP CONSTRAINT sweep_results_sweep_id_sample_id_voice_key,
            DROP COLUMN voice,
            ADD UNIQUE (sweep_id, sample_id);
        """
    )
