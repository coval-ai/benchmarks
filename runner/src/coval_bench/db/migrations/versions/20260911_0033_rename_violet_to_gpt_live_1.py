# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Rename ``violet`` to ``openai/gpt-live-1``; register it and ``xai/grok-voice`` as early access.

``violet`` was the pre-launch label for the GPT-Live-1 agents on the three
instruction-adherence industry boards. GPT-Live-1 launched publicly on
2026-07-08, so the codename no longer protects anything and just reads as an
unknown model on the dashboard.

``model`` is stored on every results row and is part of the bucket tables'
primary keys, so every table keyed by (provider, model) is rewritten together.
Rewriting only some would split the industry series between two keys.

Neither the codename nor ``xai/grok-voice`` was ever registered, so the industry
boards served both to the public: an unregistered pair cannot be embargoed. Both
get registry rows here as early access (``published = FALSE``), which is what
pulls them back behind the gate while giving them a name and a color for the orgs
that may see them.

The per-window matviews are deliberately not refreshed here: the runner
refreshes them at the end of each benchmark run (see ``migrations/env.py``),
and ``REFRESH ... CONCURRENTLY`` cannot run inside a migration's transaction.
"""

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
    """Point every violet row at gpt-live-1 and register both models as early access."""
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
    """Restore the codename and drop both registry rows (tags cascade)."""
    op.execute(
        "DELETE FROM benchmarks_v2.models "
        "WHERE modality = 'S2S' AND updated_by_user_id = 'migration:20260911_0033' "
        "AND (provider, model) IN (('openai', 'gpt-live-1'), ('xai', 'grok-voice'))"
    )
    _rename("gpt-live-1", "violet")
