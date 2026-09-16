# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Move LLM request configuration into the registry and register GPT-5 variants."""

from alembic import op

revision = "20260916_0037"
down_revision = "20260915_0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE benchmarks_v2.models ADD COLUMN llm_config jsonb
            CHECK (llm_config IS NULL OR
                (modality = 'LLM' AND jsonb_typeof(llm_config) = 'object'));

        WITH previous AS MATERIALIZED (
            SELECT id, to_jsonb(m) - 'updated_by_email' AS snapshot
            FROM benchmarks_v2.models m
        ), changed AS (
            UPDATE benchmarks_v2.models m
            SET llm_config = jsonb_build_object(
                    'upstream_model', m.model,
                    'reasoning_effort', CASE WHEN m.provider = 'google' THEN 'none' END,
                    'legacy_provider_route', true),
                updated_at = now(), updated_by_user_id = 'migration:20260916_0037'
            WHERE modality = 'LLM' AND (provider, model) IN (
                ('phonely', 'phonely-agent'), ('openai', 'gpt-4.1'),
                ('google', 'gemini-2.5-flash'))
            RETURNING m.*
        )
        INSERT INTO benchmarks_v2.model_history
            (model_id, modality, provider, model, old, new, changed_by_user_id)
        SELECT changed.id, modality, provider, model, previous.snapshot,
               to_jsonb(changed) - 'updated_by_email', 'migration:20260916_0037'
        FROM changed JOIN previous ON previous.id = changed.id;

        CREATE UNIQUE INDEX models_llm_legacy_provider_route
            ON benchmarks_v2.models (provider)
            WHERE modality = 'LLM' AND llm_config->>'legacy_provider_route' = 'true';

        WITH added AS (
            INSERT INTO benchmarks_v2.models
                (modality, provider, model, creator, source, licensing, region,
                 arena_enabled, collected, published, llm_config, updated_by_user_id)
            SELECT 'LLM', 'openai', 'gpt-5-' || effort, 'openai', 'official-api',
                   'proprietary', 'us', false, true, false,
                   jsonb_build_object('upstream_model', 'gpt-5',
                                      'reasoning_effort', effort,
                                      'legacy_provider_route', false),
                   'migration:20260916_0037'
            FROM (VALUES ('minimal'), ('medium')) AS variants(effort)
            ON CONFLICT (modality, provider, model) DO NOTHING
            RETURNING *
        )
        INSERT INTO benchmarks_v2.model_history
            (model_id, modality, provider, model, old, new, changed_by_user_id)
        SELECT id, modality, provider, model, NULL,
               to_jsonb(added) - 'updated_by_email', 'migration:20260916_0037'
        FROM added;
    """)


def downgrade() -> None:
    # Remove only unedited seed entries. Results use provider/model keys and are
    # retained. Restore provenance on compatibility rows for earlier downgrades.
    op.execute("""
        DELETE FROM benchmarks_v2.models
        WHERE updated_by_user_id = 'migration:20260916_0037'
          AND modality = 'LLM' AND provider = 'openai'
          AND model IN ('gpt-5-minimal', 'gpt-5-medium');
        UPDATE benchmarks_v2.models m
        SET updated_by_user_id = h.old->>'updated_by_user_id',
            updated_at = (h.old->>'updated_at')::timestamptz
        FROM benchmarks_v2.model_history h
        WHERE m.id = h.model_id AND h.old IS NOT NULL
          AND h.changed_by_user_id = 'migration:20260916_0037'
          AND m.updated_by_user_id = 'migration:20260916_0037';
        DELETE FROM benchmarks_v2.model_history
        WHERE changed_by_user_id = 'migration:20260916_0037';
        DROP INDEX benchmarks_v2.models_llm_legacy_provider_route;
        ALTER TABLE benchmarks_v2.models DROP COLUMN llm_config;
    """)
