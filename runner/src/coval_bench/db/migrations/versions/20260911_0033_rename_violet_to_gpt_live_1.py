# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rename Violet to GPT-Live-1 and register GPT-Live-1 and Grok Voice as early access."""

from __future__ import annotations

from alembic import op

revision = "20260911_0033"
down_revision = "20260910_0032"
branch_labels = None
depends_on = None

_RENAME_ROWS = """
UPDATE benchmarks_v2.results SET model = '{new}'
    WHERE provider = 'openai' AND model = '{old}';
UPDATE benchmarks_v2.results_by_bucket SET model = '{new}'
    WHERE provider = 'openai' AND model = '{old}';
UPDATE benchmarks_v2.metric_values_by_bucket SET model = '{new}'
    WHERE provider = 'openai' AND model = '{old}';
UPDATE benchmarks_v2.benchmark_observations SET model = '{new}'
    WHERE provider = 'openai' AND model = '{old}';
"""


def _rename(old: str, new: str) -> None:
    op.execute(_RENAME_ROWS.format(old=old, new=new))  # noqa: S608 — both values are literals above


def upgrade() -> None:
    _rename("violet", "gpt-live-1")
    op.execute(
        """
        INSERT INTO benchmarks_v2.models
            (modality, provider, model, voice, voices, creator, source, licensing,
             on_prem, region, arena_enabled, collected, published, color, updated_by_user_id)
        VALUES
            ('S2S', 'openai', 'gpt-live-1', NULL, '[]'::jsonb, NULL, 'official-api', 'proprietary',
             FALSE, 'us', TRUE, TRUE, FALSE, '#e08bb5', 'migration:20260911_0033'),
            ('S2S', 'xai', 'grok-voice', NULL, '[]'::jsonb, NULL, 'official-api', 'proprietary',
             FALSE, 'us', TRUE, TRUE, FALSE, '#9a6ad1', 'migration:20260911_0033')
        ON CONFLICT (modality, provider, model) DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO benchmarks_v2.model_tags (model_id, tag)
        SELECT m.id, 'multilingual'
        FROM benchmarks_v2.models m
        WHERE m.modality = 'S2S' AND m.provider = 'openai' AND m.model = 'gpt-live-1'
        ON CONFLICT DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM benchmarks_v2.models "
        "WHERE modality = 'S2S' AND updated_by_user_id = 'migration:20260911_0033' "
        "AND (provider, model) IN (('openai', 'gpt-live-1'), ('xai', 'grok-voice'))"
    )
    _rename("gpt-live-1", "violet")
