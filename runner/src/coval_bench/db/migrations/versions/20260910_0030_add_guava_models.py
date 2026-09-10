from __future__ import annotations

from alembic import op

revision = "20260910_0030"
down_revision = "20260909_0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        INSERT INTO benchmarks_v2.models
            (modality, provider, model, voice, creator, source, licensing,
             on_prem, region, arena_enabled, collected, published, updated_by_user_id)
        VALUES
            ('STT', 'guava', 'daytona-stt', NULL, 'guava', 'dedicated-inference',
             'open-weight', TRUE, 'us', FALSE, FALSE, FALSE, 'migration:20260910_0030'),
            ('TTS', 'guava', 'daytona-tts', 'grace', 'guava', 'dedicated-inference',
             'open-weight', TRUE, NULL, FALSE, FALSE, FALSE, 'migration:20260910_0030')
        ON CONFLICT (modality, provider, model) DO NOTHING;

        INSERT INTO benchmarks_v2.model_tags (model_id, tag)
        SELECT id, 'vad' FROM benchmarks_v2.models
        WHERE modality = 'STT' AND provider = 'guava' AND model = 'daytona-stt'
          AND updated_by_user_id = 'migration:20260910_0030'
        ON CONFLICT DO NOTHING;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DELETE FROM benchmarks_v2.models
        WHERE updated_by_user_id = 'migration:20260910_0030'
          AND provider = 'guava'
          AND (modality, model) IN (('STT', 'daytona-stt'), ('TTS', 'daytona-tts'));
        """
    )
