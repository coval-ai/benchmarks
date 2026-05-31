# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider on/off matrix — per ADR-011.

Encode the full provider × model matrix here, NOT in the orchestrator.

TODO: support env-var override via ``PROVIDER_OVERRIDES_JSON`` to flip flags
at deploy time without a code change.
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
# STT matrix
# Uncommented blocks → enabled=True; commented-out blocks → enabled=False
# ---------------------------------------------------------------------------

DEFAULT_STT_MATRIX: list[ProviderEntry] = [
    ProviderEntry(provider="deepgram", model="nova-2", enabled=True),
    ProviderEntry(provider="deepgram", model="nova-3", enabled=True),
    ProviderEntry(provider="deepgram", model="flux-general-en", enabled=True),
    ProviderEntry(provider="deepgram", model="flux-general-multi", enabled=True),
    ProviderEntry(provider="elevenlabs", model="scribe_v2_realtime", enabled=True),
    ProviderEntry(provider="assemblyai", model="universal-streaming", enabled=True),
    ProviderEntry(provider="speechmatics", model="default", enabled=True),
    ProviderEntry(provider="speechmatics", model="enhanced", enabled=True),
    ProviderEntry(provider="gradium", model="default", enabled=True),
    # OFF; disabled=True hides these from the public catalogue.
    ProviderEntry(provider="google", model="short", enabled=False, disabled=True),
    ProviderEntry(provider="google", model="long", enabled=False, disabled=True),
    ProviderEntry(provider="google", model="telephony", enabled=False, disabled=True),
    ProviderEntry(provider="google", model="chirp_2", enabled=False, disabled=True),
]

# ---------------------------------------------------------------------------
# TTS matrix
# Uncommented blocks → enabled=True; commented-out blocks → enabled=False
# ---------------------------------------------------------------------------

DEFAULT_TTS_MATRIX: list[ProviderEntry] = [
    # ElevenLabs — already in production.
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
    # OpenAI HTTP models.
    ProviderEntry(provider="openai", model="gpt-4o-mini-tts", voice="alloy", enabled=True),
    # Legacy OpenAI HTTP models — retired 2026-05-31; kept disabled (hidden from the
    # public catalogue) per the matrix convention for superseded models.
    ProviderEntry(provider="openai", model="tts-1-hd", voice="alloy", enabled=False, disabled=True),
    ProviderEntry(provider="openai", model="tts-1", voice="alloy", enabled=False, disabled=True),
    # Cartesia — re-activated 2026-04-30: sonic-3 (flagship).
    ProviderEntry(
        provider="cartesia",
        model="sonic-3",
        voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        enabled=True,
    ),
    # Deepgram — re-activated 2026-04-30: aura-2-thalia-en.
    ProviderEntry(
        provider="deepgram",
        model="aura-2-thalia-en",
        voice="aura-2-thalia-en",
        enabled=True,
    ),
    # Gradium — added 2026-05-03: WebSocket streaming, Emma voice (American English).
    ProviderEntry(
        provider="gradium",
        model="default",
        voice="YTpq7expH9539ERJ",
        enabled=True,
    ),
    # Rime — all three on /ws3 WebSocket (rewired 2026-05-19).
    # "arcana" resolves server-side to Arcana v3; "coda" to May 2026 flagship.
    ProviderEntry(provider="rime", model="coda", voice="luna", enabled=True),
    ProviderEntry(provider="rime", model="arcana", voice="luna", enabled=True),
    ProviderEntry(provider="rime", model="mistv3", voice="luna", enabled=True),
    ProviderEntry(provider="rime", model="mistv2", voice="luna", enabled=False, disabled=True),
    ProviderEntry(
        provider="hume",
        model="octave-tts",
        voice="176a55b1-4468-4736-8878-db82729667c1",
        enabled=True,
    ),
    ProviderEntry(
        provider="hume",
        model="octave-2",
        voice="176a55b1-4468-4736-8878-db82729667c1",
        enabled=True,
    ),
    # Placeholder entries with voice=None — never executed by the runner; surfaced only
    # in /v1/providers so the frontend can label them as disabled.
    ProviderEntry(
        provider="openai",
        model="gpt-realtime-2025-08-28",
        voice=None,
        enabled=False,
        disabled=True,
    ),
    ProviderEntry(provider="cartesia", model="sonic", voice=None, enabled=False, disabled=True),
]
