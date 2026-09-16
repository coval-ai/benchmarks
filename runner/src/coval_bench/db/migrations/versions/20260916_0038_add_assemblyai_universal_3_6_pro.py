# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Register AssemblyAI universal-3.6-pro (STT), not yet collected.

Revision ID: 20260916_0038
Revises:     20260916_0037
Create Date: 2026-09-11

Registered with ``collected`` and ``published`` both off, the way new models
land here: the row exists so the admin registry can flip ``collected`` on after
release, once the benchmark's key is confirmed to have access, and ``published``
when results should show. It carries the same feature tags as the
universal-3.5-pro row it sits beside. No pricing row — the vendor has not published a rate — and
``color`` stays NULL so the site picks one until a color is recorded.
"""

from __future__ import annotations

from alembic import op

revision = "20260916_0038"
down_revision = "20260916_0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        INSERT INTO benchmarks_v2.models
            (modality, provider, model, creator, source, licensing,
             on_prem, region, arena_enabled, collected, published, updated_by_user_id)
        VALUES
            ('STT', 'assemblyai', 'universal-3.6-pro', NULL, 'official-api', 'proprietary',
             TRUE, 'us', FALSE, FALSE, FALSE, 'migration:20260916_0038')
        ON CONFLICT (modality, provider, model) DO NOTHING;

        INSERT INTO benchmarks_v2.model_tags (model_id, tag)
        SELECT m.id, t.tag
        FROM benchmarks_v2.models AS m
        CROSS JOIN (VALUES
            ('code-switching'), ('diarization'), ('keyterm-biasing'), ('multilingual'), ('vad')
        ) AS t(tag)
        WHERE m.modality = 'STT' AND m.provider = 'assemblyai' AND m.model = 'universal-3.6-pro'
          AND m.updated_by_user_id = 'migration:20260916_0038'
        ON CONFLICT DO NOTHING;
        """
    )


def downgrade() -> None:
    # model_tags rows go with the model (ON DELETE CASCADE).
    op.execute(
        """
        DELETE FROM benchmarks_v2.models
        WHERE updated_by_user_id = 'migration:20260916_0038'
          AND modality = 'STT' AND provider = 'assemblyai' AND model = 'universal-3.6-pro';
        """
    )
