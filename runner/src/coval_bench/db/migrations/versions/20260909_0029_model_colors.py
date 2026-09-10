# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Seed rows are one line each so the palette reads as a table.

"""Per-model series colors: models.color, seeded with what the site draws today.

Revision ID: 20260909_0029
Revises:     20260909_0028
Create Date: 2026-09-09

The site used to pick every model's color from a palette compiled into the frontend;
recoloring a model meant a deploy. The color now lives on the model row, edited from
the admin registry like any other field, and ``/v1/providers`` carries it so the
frontend draws what the registry says.

The seed is the color the site was drawing each registered model in as of this
revision — read off the live palette and pasted here as a literal, so the public
pages look exactly the same the day the column lands and the registry, not the
site's code, is the source of truth from then on.
NULL means the site still picks: a model registered after this seed keeps the
compiled palette until someone records a color for it.

``color`` is nullable text constrained to lowercase ``#rrggbb``; case is normalized
at the API boundary. The ``api`` role's existing GRANTs on ``models`` cover it.
"""

from __future__ import annotations

from alembic import op

revision = "20260909_0029"
down_revision = "20260909_0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the color column and seed it with the palette the site draws today."""
    op.execute(
        """
        ALTER TABLE benchmarks_v2.models
            ADD COLUMN color TEXT CHECK (color IS NULL OR color ~ '^#[0-9a-f]{6}$');
        """
    )
    op.get_bind().exec_driver_sql(
        """
        UPDATE benchmarks_v2.models m
        SET color = v.color
        FROM (VALUES
            ('LLM', 'phonely', 'phonely-agent', '#c6668d'),
            ('S2S', 'colors', 'gray', '#4b9ac0'),
            ('S2S', 'colors', 'red', '#4b9ac0'),
            ('S2S', 'google', 'gemini-live', '#53b0de'),
            ('S2S', 'openai', 'gpt-realtime', '#c35483'),
            ('S2S', 'xai', 'grok-voice-think-fast-1.0', '#a879c5'),
            ('S2S', 'xai', 'grok-voice-think-fast-2.0', '#c890ea'),
            ('STT', 'assemblyai', 'universal-3-pro', '#a871c9'),
            ('STT', 'assemblyai', 'universal-3.5-pro', '#bf86e0'),
            ('STT', 'assemblyai', 'universal-streaming', '#965fb6'),
            ('STT', 'assemblyai', 'universal-streaming-multilingual', '#d299f5'),
            ('STT', 'azure', 'default', '#3d7fd1'),
            ('STT', 'baseten', 'qwen3-asr-1.7b', '#15c25c'),
            ('STT', 'baseten', 'whisper-large-v3', '#19e76e'),
            ('STT', 'cartesia', 'ink-2', '#3b6eb9'),
            ('STT', 'deepgram', 'flux-general-en', '#039580'),
            ('STT', 'deepgram', 'flux-general-multi', '#007b69'),
            ('STT', 'deepgram', 'nova-2', '#49cdb4'),
            ('STT', 'deepgram', 'nova-3', '#1db098'),
            ('STT', 'elevenlabs', 'scribe_v2_realtime', '#e67e54'),
            ('STT', 'gemini', 'gemini-3.5-transcribe-live', '#3cc3ab'),
            ('STT', 'gladia', 'solaria-1', '#6a9fee'),
            ('STT', 'google', 'chirp_2', '#53b0de'),
            ('STT', 'google', 'chirp_3', '#1d85b0'),
            ('STT', 'gradium', 'default', '#74c16c'),
            ('STT', 'inworld', 'inworld-stt-1', '#736dc3'),
            ('STT', 'mistral', 'voxtral-mini-transcribe-realtime-2602', '#835a01'),
            ('STT', 'modulate', 'english-fast-transcription-streaming', '#3cc3ab'),
            ('STT', 'modulate', 'multilingual-transcription-streaming', '#5bcdb8'),
            ('STT', 'modulate', 'velma-2-stt-streaming', '#5bcdb8'),
            ('STT', 'modulate', 'velma-2-stt-streaming-english-v2', '#32a490'),
            ('STT', 'openai', 'gpt-4o-mini-transcribe', '#d1608f'),
            ('STT', 'openai', 'gpt-4o-transcribe', '#f782b1'),
            ('STT', 'openai', 'gpt-realtime-whisper', '#b24574'),
            ('STT', 'reson8', 'realtime', '#8e9f36'),
            ('STT', 'revai', 'reverb', '#7f7bc6'),
            ('STT', 'smallest', 'pulse', '#b07b05'),
            ('STT', 'soniox', 'stt-rt-v4', '#707f03'),
            ('STT', 'soniox', 'stt-rt-v5', '#697800'),
            ('STT', 'speechmatics', 'default', '#a24112'),
            ('STT', 'speechmatics', 'enhanced', '#c35f36'),
            ('STT', 'speechmatics', 'linden-1', '#cd7956'),
            ('STT', 'together', 'nemotron-3-asr-streaming-0.6b', '#4a9243'),
            ('STT', 'together', 'nemotron-3.5-asr-streaming-0.6b', '#4a9243'),
            ('STT', 'together', 'parakeet-tdt-0.6b-v3', '#33792c'),
            ('STT', 'together', 'whisper-large-v3', '#8f3055'),
            ('STT', 'xai', 'grok-stt', '#844da2'),
            ('STT', 'zoom', 'scribe', '#5bcdb8'),
            ('TTS', 'alibaba', 'qwen3-tts-flash-realtime', '#9c6c03'),
            ('TTS', 'atlas', 'atlas-tts', '#3cc3ab'),
            ('TTS', 'azure', 'dragon-hd-latest', '#3d7fd1'),
            ('TTS', 'azure', 'neural', '#5c93d8'),
            ('TTS', 'baseten', 'qwen3-tts-1.7b', '#19e76e'),
            ('TTS', 'cartesia', 'sonic', '#4f7bbd'),
            ('TTS', 'cartesia', 'sonic-3', '#5e93e1'),
            ('TTS', 'cartesia', 'sonic-3.5', '#5e93e1'),
            ('TTS', 'cartesia', 'sonic-3.6', '#5e93e1'),
            ('TTS', 'cartesia', 'sonic-preview', '#4f7bbd'),
            ('TTS', 'deepdub', 'dd-etts-3.3', '#5bcdb8'),
            ('TTS', 'deepgram', 'aura-2-thalia-en', '#29b69e'),
            ('TTS', 'deepgram', 'flux-haley-en', '#189480'),
            ('TTS', 'elevenlabs', 'eleven_flash_v2_5', '#d87248'),
            ('TTS', 'elevenlabs', 'eleven_multilingual_v2', '#fb9167'),
            ('TTS', 'elevenlabs', 'eleven_turbo_v2_5', '#bd592f'),
            ('TTS', 'elevenlabs', 'eleven_v3', '#b04d20'),
            ('TTS', 'elevenlabs', 'eleven_v3_conversational', '#c16a47'),
            ('TTS', 'fishaudio', 's1', '#ef8eb6'),
            ('TTS', 'fishaudio', 's2.1-pro', '#c6668d'),
            ('TTS', 'fishaudio', 's2.1-pro-free', '#ec79a8'),
            ('TTS', 'fluxions', 'vui', '#f78e64'),
            ('TTS', 'google', 'chirp-3-hd', '#4694ba'),
            ('TTS', 'google', 'gemini-2.5-flash-tts', '#53b0de'),
            ('TTS', 'gradium', 'default', '#74c16c'),
            ('TTS', 'gradium', 'gradium-tts-beta', '#74c16c'),
            ('TTS', 'groq', 'canopylabs/orpheus-v1-english', '#dea645'),
            ('TTS', 'hakim', 'hakim-fast-v1', '#cc9ce8'),
            ('TTS', 'hume', 'octave-2', '#59b7e4'),
            ('TTS', 'hume', 'octave-tts', '#268bb6'),
            ('TTS', 'inworld', 'inworld-tts-1.5-max', '#8b85de'),
            ('TTS', 'inworld', 'inworld-tts-1.5-mini', '#a9a5ff'),
            ('TTS', 'inworld', 'inworld-tts-2', '#5b55a9'),
            ('TTS', 'inworld', 'inworld-tts-2-flash', '#4c4294'),
            ('TTS', 'lmnt', 'blizzard', '#3cc3ab'),
            ('TTS', 'minimax', 'speech-2.8-hd', '#3cc3ab'),
            ('TTS', 'minimax', 'speech-2.8-turbo', '#32a490'),
            ('TTS', 'murf', 'falcon-2', '#a373c0'),
            ('TTS', 'openai', 'gpt-4o-mini-tts', '#de6d9b'),
            ('TTS', 'openai', 'gpt-realtime-2025-08-28', '#d1608f'),
            ('TTS', 'openai', 'tts-1', '#b05178'),
            ('TTS', 'openai', 'tts-1-hd', '#b05178'),
            ('TTS', 'palabra', 'palabra-tts-v1', '#ba8b3a'),
            ('TTS', 'rime', 'arcana', '#55a14e'),
            ('TTS', 'rime', 'coda', '#3c8836'),
            ('TTS', 'rime', 'mistv2', '#32722d'),
            ('TTS', 'rime', 'mistv3', '#22701c'),
            ('TTS', 'smallest', 'lightning_v3.1_pro', '#d7a03d'),
            ('TTS', 'soniox', 'tts-rt-v1', '#9db030'),
            ('TTS', 'soniox', 'tts-rt-v2', '#859704'),
            ('TTS', 'speechify', 'simba-3.0', '#64a55d'),
            ('TTS', 'speechify', 'simba-3.2', '#8dcd86'),
            ('TTS', 'xai', 'grok-tts', '#c890ea')
        ) AS v (modality, provider, model, color)
        WHERE m.modality = v.modality AND m.provider = v.provider AND m.model = v.model;
        """
    )


def downgrade() -> None:
    """Drop the color column; the site falls back to its compiled palette."""
    op.execute("ALTER TABLE benchmarks_v2.models DROP COLUMN color;")
