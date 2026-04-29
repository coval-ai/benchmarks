# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider on/off matrix — per ADR-011.

Encode the full provider × model matrix here, NOT in the orchestrator.
Defaults mirror the legacy ``run_stt.py`` / ``run_tts.py`` enabled/commented state.

TODO: Phase 3 — support env-var override via ``PROVIDER_OVERRIDES_JSON`` to flip
flags at deploy time without a code change.
"""

from __future__ import annotations

from pydantic import BaseModel


class ProviderEntry(BaseModel):
    """A single provider × model × voice configuration entry."""

    provider: str
    model: str
    voice: str | None = None  # TTS only
    enabled: bool
    disabled: bool = False  # admin-level "hide / don't run" flag; orthogonal to enabled


# ---------------------------------------------------------------------------
# STT matrix — mirrors legacy run_stt.py CONFIGURATIONS dict
# Uncommented blocks → enabled=True; commented-out blocks → enabled=False
# ---------------------------------------------------------------------------

DEFAULT_STT_MATRIX: list[ProviderEntry] = [
    ProviderEntry(provider="deepgram", model="nova-2", enabled=True),
    ProviderEntry(provider="deepgram", model="nova-3", enabled=True),
    ProviderEntry(provider="deepgram", model="flux-general-en", enabled=True),
    ProviderEntry(provider="elevenlabs", model="scribe_v2_realtime", enabled=True),
    ProviderEntry(provider="assemblyai", model="universal-streaming", enabled=True),
    ProviderEntry(provider="speechmatics", model="default", enabled=True),
    ProviderEntry(provider="speechmatics", model="enhanced", enabled=True),
    # OFF per legacy comments; disabled=True mirrors legacy EXCLUDED_MODELS in
    # benchmarking-web-app/lib/db.ts
    ProviderEntry(provider="google", model="short", enabled=False, disabled=True),
    ProviderEntry(provider="google", model="long", enabled=False, disabled=True),
    ProviderEntry(provider="google", model="telephony", enabled=False, disabled=True),
    ProviderEntry(provider="google", model="chirp_2", enabled=False, disabled=True),
]

# ---------------------------------------------------------------------------
# TTS matrix — mirrors legacy run_tts.py CONFIGURATIONS dict
# Uncommented blocks → enabled=True; commented-out blocks → enabled=False
# ---------------------------------------------------------------------------

DEFAULT_TTS_MATRIX: list[ProviderEntry] = [
    ProviderEntry(
        provider="elevenlabs",
        model="eleven_flash_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
        enabled=True,
    ),
    ProviderEntry(
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        voice="IKne3meq5aSn9XLyUdCD",
        enabled=True,
    ),
    ProviderEntry(
        provider="elevenlabs",
        model="eleven_turbo_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
        enabled=True,
    ),
    # OFF per legacy comments:
    ProviderEntry(provider="openai", model="tts-1", voice="alloy", enabled=False),
    ProviderEntry(provider="openai", model="gpt-4o-mini-tts", voice="alloy", enabled=False),
    ProviderEntry(
        provider="cartesia",
        model="sonic-3",
        voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        enabled=False,
    ),
    ProviderEntry(
        provider="deepgram",
        model="aura-2-thalia-en",
        voice="aura-2-thalia-en",
        enabled=False,
    ),
    # disabled=True mirrors legacy EXCLUDED_MODELS in benchmarking-web-app/lib/db.ts
    ProviderEntry(
        provider="hume", model="octave-tts", voice="male_01", enabled=False, disabled=True
    ),
    ProviderEntry(provider="hume", model="octave-2", voice="male_01", enabled=False, disabled=True),
    ProviderEntry(provider="rime", model="arcana", voice="luna", enabled=False),
    ProviderEntry(provider="rime", model="mistv2", voice="luna", enabled=False),
    # Placeholder entries for models in legacy EXCLUDED_MODELS that were never in this matrix.
    # voice=None because we never run them — purpose is to expose them in /v1/providers so the
    # frontend can label them as disabled. Both came from TTS legacy data (web-app-data-audit.md).
    ProviderEntry(
        provider="openai",
        model="gpt-realtime-2025-08-28",
        voice=None,
        enabled=False,
        disabled=True,
    ),
    ProviderEntry(provider="cartesia", model="sonic", voice=None, enabled=False, disabled=True),
]
