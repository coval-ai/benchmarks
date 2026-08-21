# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Move model lifecycle state out of the registry into the database.

``model_state`` holds the two booleans that used to be the registry's
``status`` field: ``running`` (the orchestrator schedules the model) and
``shown`` (the public site shows it). One row per registry model, keyed by the
registry's natural key. A registry entry with no row defaults to running and
not shown (the old EARLY_ACCESS), so a newly merged model starts benchmarking
under embargo with nothing public until someone flips it in the admin page.

``model_state_history`` records every change — who, when, from what, to what —
so a gap in a model's data can be traced to the toggle that caused it. Old
values are NULL when no row existed before the change (the seed below, or the
first toggle of a model added to the registry later).

The seed is a literal snapshot of the registry statuses at authoring time
(ACTIVE -> running+shown, PAUSED -> shown only, EARLY_ACCESS -> running only,
RETIRED/PENDING -> neither), not a registry import: the ``status`` field is
deleted from the registry in the same change, and this file has to keep
working after that.
"""

from __future__ import annotations

from alembic import op

revision = "20260820_0019"
down_revision = "20260818_0018"
branch_labels = None
depends_on = None

# (benchmark, provider, model, running, shown) — registry snapshot, 2026-08-20.
_SEED_VALUES = """
    ('STT', 'deepgram', 'nova-2', true, true),
    ('STT', 'deepgram', 'nova-3', true, true),
    ('STT', 'deepgram', 'flux-general-en', true, true),
    ('STT', 'deepgram', 'flux-general-multi', true, true),
    ('STT', 'elevenlabs', 'scribe_v2_realtime', true, true),
    ('STT', 'openai', 'gpt-realtime-whisper', true, true),
    ('STT', 'openai', 'gpt-4o-transcribe', true, true),
    ('STT', 'openai', 'gpt-4o-mini-transcribe', true, true),
    ('STT', 'assemblyai', 'universal-streaming', true, true),
    ('STT', 'assemblyai', 'universal-streaming-multilingual', false, false),
    ('STT', 'assemblyai', 'universal-3-pro', false, false),
    ('STT', 'assemblyai', 'universal-3.5-pro', true, true),
    ('STT', 'speechmatics', 'default', true, true),
    ('STT', 'speechmatics', 'enhanced', true, true),
    ('STT', 'gradium', 'default', true, true),
    ('STT', 'gladia', 'solaria-1', true, true),
    ('STT', 'soniox', 'stt-rt-v4', false, false),
    ('STT', 'soniox', 'stt-rt-v5', true, true),
    ('STT', 'inworld', 'inworld-stt-1', true, true),
    ('STT', 'xai', 'grok-stt', true, true),
    ('STT', 'smallest', 'pulse', true, true),
    ('STT', 'cartesia', 'ink-2', true, true),
    ('STT', 'mistral', 'voxtral-mini-transcribe-realtime-2602', true, true),
    ('STT', 'baseten', 'whisper-large-v3', true, false),
    ('STT', 'azure', 'default', false, true),
    ('STT', 'google', 'chirp_2', true, true),
    ('STT', 'google', 'chirp_3', true, true),
    ('STT', 'revai', 'reverb', false, true),
    ('STT', 'together', 'nemotron-3-asr-streaming-0.6b', false, false),
    ('STT', 'together', 'nemotron-3.5-asr-streaming-0.6b', true, true),
    ('STT', 'together', 'parakeet-tdt-0.6b-v3', true, true),
    ('STT', 'together', 'whisper-large-v3', true, true),
    ('STT', 'reson8', 'realtime', true, true),
    ('STT', 'modulate', 'velma-2-stt-streaming-english-v2', false, false),
    ('STT', 'modulate', 'velma-2-stt-streaming', false, true),
    ('STT', 'modulate', 'english-fast-transcription-streaming', false, false),
    ('STT', 'modulate', 'multilingual-transcription-streaming', false, false),
    ('TTS', 'elevenlabs', 'eleven_flash_v2_5', false, false),
    ('TTS', 'elevenlabs', 'eleven_multilingual_v2', false, false),
    ('TTS', 'elevenlabs', 'eleven_turbo_v2_5', false, false),
    ('TTS', 'elevenlabs', 'eleven_v3', false, false),
    ('TTS', 'elevenlabs', 'eleven_v3_conversational', true, false),
    ('TTS', 'openai', 'gpt-4o-mini-tts', true, true),
    ('TTS', 'openai', 'tts-1-hd', false, false),
    ('TTS', 'openai', 'tts-1', false, false),
    ('TTS', 'cartesia', 'sonic-3', false, false),
    ('TTS', 'cartesia', 'sonic-3.5', true, true),
    ('TTS', 'cartesia', 'sonic-preview', true, false),
    ('TTS', 'deepgram', 'aura-2-thalia-en', true, true),
    ('TTS', 'deepgram', 'flux-haley-en', true, false),
    ('TTS', 'gradium', 'default', true, true),
    ('TTS', 'palabra', 'palabra-tts-v1', true, true),
    ('TTS', 'rime', 'coda', true, true),
    ('TTS', 'rime', 'arcana', false, false),
    ('TTS', 'rime', 'mistv3', true, true),
    ('TTS', 'rime', 'mistv2', false, false),
    ('TTS', 'hume', 'octave-tts', false, false),
    ('TTS', 'hume', 'octave-2', false, false),
    ('TTS', 'xai', 'grok-tts', true, true),
    ('TTS', 'smallest', 'lightning_v3.1_pro', true, true),
    ('TTS', 'inworld', 'inworld-tts-2', true, true),
    ('TTS', 'inworld', 'inworld-tts-2-flash', true, true),
    ('TTS', 'inworld', 'inworld-tts-1.5-max', false, false),
    ('TTS', 'inworld', 'inworld-tts-1.5-mini', false, false),
    ('TTS', 'soniox', 'tts-rt-v1', true, true),
    ('TTS', 'soniox', 'tts-rt-v2', true, true),
    ('TTS', 'azure', 'neural', false, true),
    ('TTS', 'azure', 'dragon-hd-latest', false, true),
    ('TTS', 'groq', 'canopylabs/orpheus-v1-english', false, false),
    ('TTS', 'google', 'chirp-3-hd', true, true),
    ('TTS', 'google', 'gemini-2.5-flash-tts', false, false),
    ('TTS', 'baseten', 'qwen3-tts-1.7b', true, false),
    ('TTS', 'alibaba', 'qwen3-tts-flash-realtime', true, true),
    ('TTS', 'fishaudio', 's1', true, true),
    ('TTS', 'fishaudio', 's2.1-pro', true, true),
    ('TTS', 'fishaudio', 's2.1-pro-free', true, true),
    ('TTS', 'minimax', 'speech-2.8-hd', false, true),
    ('TTS', 'minimax', 'speech-2.8-turbo', false, true),
    ('TTS', 'speechify', 'simba-3.2', true, true),
    ('TTS', 'speechify', 'simba-3.0', true, true),
    ('TTS', 'fluxions', 'vui', true, true),
    ('TTS', 'lmnt', 'blizzard', true, true),
    ('TTS', 'deepdub', 'dd-etts-3.3', true, false),
    ('TTS', 'murf', 'falcon-2', true, true),
    ('TTS', 'hakim', 'hakim-fast-v1', true, false),
    ('TTS', 'openai', 'gpt-realtime-2025-08-28', false, false),
    ('TTS', 'cartesia', 'sonic', false, false),
    ('S2S', 'openai', 'gpt-realtime', true, true),
    ('S2S', 'google', 'gemini-live', true, true),
    ('S2S', 'xai', 'grok-voice-think-fast-1.0', true, false),
    ('S2S', 'xai', 'grok-voice-think-fast-2.0', true, false),
    ('S2S', 'colors', 'gray', true, false),
    ('S2S', 'colors', 'red', true, false)
"""

_SEEDED_BY = "seed:20260820_0019"


def upgrade() -> None:
    """Create the state tables and seed them from the registry snapshot."""
    op.execute("""
        CREATE TABLE benchmarks_v2.model_state (
            benchmark   text NOT NULL CHECK (benchmark IN ('STT','TTS','S2S')),
            provider    text NOT NULL,
            model       text NOT NULL,
            running     boolean NOT NULL,
            shown       boolean NOT NULL,
            updated_by  text NOT NULL,
            updated_at  timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (benchmark, provider, model)
        )
    """)
    op.execute("""
        CREATE TABLE benchmarks_v2.model_state_history (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            benchmark   text NOT NULL,
            provider    text NOT NULL,
            model       text NOT NULL,
            old_running boolean,
            old_shown   boolean,
            new_running boolean NOT NULL,
            new_shown   boolean NOT NULL,
            changed_by  text NOT NULL,
            changed_at  timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX model_state_history_key_idx
            ON benchmarks_v2.model_state_history (benchmark, provider, model, changed_at DESC)
    """)
    op.execute(f"""
        INSERT INTO benchmarks_v2.model_state
            (benchmark, provider, model, running, shown, updated_by)
        SELECT v.benchmark, v.provider, v.model, v.running, v.shown, '{_SEEDED_BY}'
        FROM (VALUES {_SEED_VALUES.strip()}) AS v(benchmark, provider, model, running, shown)
    """)  # noqa: S608 — the values are the literal snapshot above, no caller input
    op.execute(f"""
        INSERT INTO benchmarks_v2.model_state_history
            (benchmark, provider, model, old_running, old_shown, new_running, new_shown,
             changed_by, changed_at)
        SELECT benchmark, provider, model, NULL, NULL, running, shown, '{_SEEDED_BY}', updated_at
        FROM benchmarks_v2.model_state
    """)  # noqa: S608 — _SEEDED_BY is the constant above


def downgrade() -> None:
    """Drop both state tables. The registry snapshot in source is the fallback."""
    op.execute("DROP TABLE benchmarks_v2.model_state_history")
    op.execute("DROP TABLE benchmarks_v2.model_state")
