# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Embedded SQL is formatted as executable migration text.

"""Publish the models that stopped being collected but stayed on the site.

Revision ID: 20260831_0024
Revises:     20260831_0023
Create Date: 2026-08-31

``published`` means the results are public; ``collected`` means we still
measure it. The seed mapped the old ``RETIRED`` status to neither, but a
retired model's results have always been public — the site lists it greyed
out. Left unpublished it would be treated as embargoed, hiding 21
models and their history from the catalogue.

The models that were never launched keep ``published = false``: nothing of
theirs has been public, and listing them would leak an unreleased model.
"""

from __future__ import annotations

from alembic import op

revision = "20260831_0024"
down_revision = "20260831_0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Publish the 21 retired models; leave the unlaunched ones alone."""
    op.get_bind().exec_driver_sql(
        """
        UPDATE benchmarks_v2.models m
        SET published = true
        FROM (VALUES
            ('STT', 'assemblyai', 'universal-streaming-multilingual'),
            ('STT', 'assemblyai', 'universal-3-pro'),
            ('STT', 'soniox', 'stt-rt-v4'),
            ('STT', 'together', 'nemotron-3-asr-streaming-0.6b'),
            ('STT', 'modulate', 'velma-2-stt-streaming-english-v2'),
            ('STT', 'modulate', 'english-fast-transcription-streaming'),
            ('STT', 'modulate', 'multilingual-transcription-streaming'),
            ('TTS', 'elevenlabs', 'eleven_multilingual_v2'),
            ('TTS', 'elevenlabs', 'eleven_turbo_v2_5'),
            ('TTS', 'elevenlabs', 'eleven_v3'),
            ('TTS', 'openai', 'tts-1-hd'),
            ('TTS', 'openai', 'tts-1'),
            ('TTS', 'cartesia', 'sonic-3'),
            ('TTS', 'rime', 'arcana'),
            ('TTS', 'rime', 'mistv2'),
            ('TTS', 'hume', 'octave-tts'),
            ('TTS', 'hume', 'octave-2'),
            ('TTS', 'inworld', 'inworld-tts-1.5-max'),
            ('TTS', 'inworld', 'inworld-tts-1.5-mini'),
            ('TTS', 'openai', 'gpt-realtime-2025-08-28'),
            ('TTS', 'cartesia', 'sonic')
        ) AS v (modality, provider, model)
        WHERE m.modality = v.modality AND m.provider = v.provider AND m.model = v.model;
        """
    )


def downgrade() -> None:
    """Return those models to unpublished."""
    op.get_bind().exec_driver_sql(
        """
        UPDATE benchmarks_v2.models m
        SET published = false
        FROM (VALUES
            ('STT', 'assemblyai', 'universal-streaming-multilingual'),
            ('STT', 'assemblyai', 'universal-3-pro'),
            ('STT', 'soniox', 'stt-rt-v4'),
            ('STT', 'together', 'nemotron-3-asr-streaming-0.6b'),
            ('STT', 'modulate', 'velma-2-stt-streaming-english-v2'),
            ('STT', 'modulate', 'english-fast-transcription-streaming'),
            ('STT', 'modulate', 'multilingual-transcription-streaming'),
            ('TTS', 'elevenlabs', 'eleven_multilingual_v2'),
            ('TTS', 'elevenlabs', 'eleven_turbo_v2_5'),
            ('TTS', 'elevenlabs', 'eleven_v3'),
            ('TTS', 'openai', 'tts-1-hd'),
            ('TTS', 'openai', 'tts-1'),
            ('TTS', 'cartesia', 'sonic-3'),
            ('TTS', 'rime', 'arcana'),
            ('TTS', 'rime', 'mistv2'),
            ('TTS', 'hume', 'octave-tts'),
            ('TTS', 'hume', 'octave-2'),
            ('TTS', 'inworld', 'inworld-tts-1.5-max'),
            ('TTS', 'inworld', 'inworld-tts-1.5-mini'),
            ('TTS', 'openai', 'gpt-realtime-2025-08-28'),
            ('TTS', 'cartesia', 'sonic')
        ) AS v (modality, provider, model)
        WHERE m.modality = v.modality AND m.provider = v.provider AND m.model = v.model;
        """
    )
