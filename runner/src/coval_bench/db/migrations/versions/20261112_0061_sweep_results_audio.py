# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep results audio.

Revision ID: 20261112_0061
Revises:     20261112_0060
Create Date: 2026-11-12

Where a TTS result's synthesized audio is kept, so a metric that needs the audio
can be scored without calling the provider again. NULL for STT.
"""

from __future__ import annotations

from alembic import op

revision = "20261112_0061"
down_revision = "20261112_0060"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE benchmarks_v2.sweep_results ADD COLUMN audio_uri TEXT CHECK (audio_uri <> '');"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE benchmarks_v2.sweep_results DROP COLUMN audio_uri;")
