# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Embedded SQL is formatted as executable migration text.

"""Sync the model tables with registry edits made after the seed.

Revision ID: 20260831_0023
Revises:     20260826_0022
Create Date: 2026-08-31
"""

from __future__ import annotations

from alembic import op

revision = "20260831_0023"
down_revision = "20260826_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Insert the post-seed models and apply the state changes."""
    op.get_bind().exec_driver_sql(
        """
        INSERT INTO benchmarks_v2.models
            (modality, provider, model, voice, voices, creator, source, licensing,
             on_prem, region, arena_enabled, collected, published, updated_by_user_id)
        VALUES
            ('STT', 'baseten', 'qwen3-asr-1.7b', NULL, '[]'::jsonb, 'alibaba', 'dedicated-inference', 'open-weight', TRUE, 'us', TRUE, TRUE, FALSE, 'migration:20260831_0023'),
            ('STT', 'gemini', 'gemini-3.5-transcribe-live', NULL, '[]'::jsonb, 'google', 'official-api', 'proprietary', FALSE, 'us', TRUE, TRUE, FALSE, 'migration:20260831_0023'),
            ('STT', 'zoom', 'scribe', NULL, '[]'::jsonb, NULL, 'official-api', 'proprietary', FALSE, 'us', TRUE, TRUE, FALSE, 'migration:20260831_0023'),
            ('TTS', 'cartesia', 'sonic-3.6', 'db6b0ed5-d5d3-463d-ae85-518a07d3c2b4', '[{"id":"db6b0ed5-d5d3-463d-ae85-518a07d3c2b4","gender":"female","name":"Skylar","accent":"en-US"},{"id":"30894953-bcce-41fe-892c-15ce19c843ff","gender":"male","name":"Parker","accent":"en-US"}]'::jsonb, NULL, 'official-api', 'proprietary', TRUE, 'us', TRUE, TRUE, TRUE, 'migration:20260831_0023'),
            ('TTS', 'atlas', 'atlas-tts', 'dax', '[]'::jsonb, NULL, 'shared-inference', 'proprietary', FALSE, 'us', FALSE, TRUE, FALSE, 'migration:20260831_0023');

        INSERT INTO benchmarks_v2.model_tags (model_id, tag)
        SELECT m.id, v.tag
        FROM benchmarks_v2.models m
        JOIN (VALUES
            ('STT', 'baseten', 'qwen3-asr-1.7b', 'multilingual'),
            ('STT', 'baseten', 'qwen3-asr-1.7b', 'vad'),
            ('STT', 'gemini', 'gemini-3.5-transcribe-live', 'keyterm-biasing'),
            ('STT', 'gemini', 'gemini-3.5-transcribe-live', 'multilingual'),
            ('STT', 'gemini', 'gemini-3.5-transcribe-live', 'vad'),
            ('STT', 'zoom', 'scribe', 'multilingual'),
            ('STT', 'zoom', 'scribe', 'vad'),
            ('TTS', 'cartesia', 'sonic-3.6', 'emotion-control'),
            ('TTS', 'cartesia', 'sonic-3.6', 'multilingual'),
            ('TTS', 'cartesia', 'sonic-3.6', 'voice-cloning')
        ) AS v (modality, provider, model, tag)
          ON v.modality = m.modality AND v.provider = m.provider AND v.model = m.model;

        UPDATE benchmarks_v2.models
        SET collected = FALSE, published = FALSE, voices = '[]'::jsonb,
            updated_by_user_id = 'migration:20260831_0023', updated_by_email = NULL,
            updated_at = now()
        WHERE modality = 'TTS' AND provider = 'cartesia' AND model = 'sonic-preview';

        UPDATE benchmarks_v2.models
        SET collected = FALSE, published = TRUE,
            updated_by_user_id = 'migration:20260831_0023', updated_by_email = NULL,
            updated_at = now()
        WHERE modality = 'TTS' AND provider = 'gradium' AND model = 'default';

        UPDATE benchmarks_v2.models
        SET collected = TRUE, published = TRUE,
            updated_by_user_id = 'migration:20260831_0023', updated_by_email = NULL,
            updated_at = now()
        WHERE modality = 'TTS' AND provider = 'gradium' AND model = 'gradium-tts-beta';
        """
    )


def downgrade() -> None:
    """Undo the sync, skipping rows the admin API has edited since."""
    op.get_bind().exec_driver_sql(
        """
        DELETE FROM benchmarks_v2.models
        WHERE updated_by_user_id = 'migration:20260831_0023'
          AND (modality, provider, model) IN (
            ('STT', 'baseten', 'qwen3-asr-1.7b'),
            ('STT', 'gemini', 'gemini-3.5-transcribe-live'),
            ('STT', 'zoom', 'scribe'),
            ('TTS', 'cartesia', 'sonic-3.6'),
            ('TTS', 'atlas', 'atlas-tts'));

        UPDATE benchmarks_v2.models
        SET collected = TRUE, published = FALSE, voices = '[{"id":"db6b0ed5-d5d3-463d-ae85-518a07d3c2b4","gender":"female","name":"Skylar","accent":"en-US"},{"id":"30894953-bcce-41fe-892c-15ce19c843ff","gender":"male","name":"Parker","accent":"en-US"}]'::jsonb,
            updated_by_user_id = 'migration:20260825_0021', updated_at = now()
        WHERE updated_by_user_id = 'migration:20260831_0023'
          AND modality = 'TTS' AND provider = 'cartesia' AND model = 'sonic-preview';

        UPDATE benchmarks_v2.models
        SET collected = TRUE, published = TRUE,
            updated_by_user_id = 'migration:20260825_0021', updated_at = now()
        WHERE updated_by_user_id = 'migration:20260831_0023'
          AND modality = 'TTS' AND provider = 'gradium' AND model = 'default';

        UPDATE benchmarks_v2.models
        SET collected = TRUE, published = FALSE,
            updated_by_user_id = 'migration:20260825_0021', updated_at = now()
        WHERE updated_by_user_id = 'migration:20260831_0023'
          AND modality = 'TTS' AND provider = 'gradium' AND model = 'gradium-tts-beta';
        """
    )
