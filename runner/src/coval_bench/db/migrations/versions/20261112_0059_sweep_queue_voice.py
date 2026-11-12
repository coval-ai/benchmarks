# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep queue voice.

Revision ID: 20261112_0059
Revises:     20261112_0058
Create Date: 2026-11-12

A TTS sweep runs every prompt in each of a model's voices, so a queued item is
a prompt in a voice. STT rows keep an empty voice.
"""

from __future__ import annotations

from alembic import op

revision = "20261112_0059"
down_revision = "20261112_0058"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE benchmarks_v2.sweep_queue
            ADD COLUMN voice TEXT NOT NULL DEFAULT '',
            DROP CONSTRAINT sweep_queue_pkey,
            ADD PRIMARY KEY (sweep_id, sample_id, voice);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DELETE FROM benchmarks_v2.sweep_queue WHERE voice <> '';
        ALTER TABLE benchmarks_v2.sweep_queue
            DROP CONSTRAINT sweep_queue_pkey,
            DROP COLUMN voice,
            ADD PRIMARY KEY (sweep_id, sample_id);
        """
    )
