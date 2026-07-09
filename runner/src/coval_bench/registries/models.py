# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry of benchmarked models — identity, run config, and status.

One entry per model, keyed by ``(benchmark, provider, model)``. The
orchestrator runs every ``ACTIVE`` entry; the API serves all entries and
marks ``RETIRED`` and ``PENDING`` ones disabled so the frontend keeps them
off the site even when historical result rows exist for them.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.tags import ModelTag


class ModelStatus(StrEnum):
    """Whether a model is benchmarked and whether the site shows it."""

    ACTIVE = "active"  # benchmarked and shown
    PAUSED = "paused"  # shown, not currently benchmarked
    RETIRED = "retired"  # not benchmarked, hidden even if old data exists
    PENDING = "pending"  # implemented but waiting on credits; hidden like retired


class Tenancy(StrEnum):
    """Whether the served model runs on shared or dedicated infrastructure."""

    SHARED = "shared"
    DEDICATED = "dedicated"


class Licensing(StrEnum):
    """Whether the model's weights are openly available."""

    PROPRIETARY = "proprietary"
    OPEN_WEIGHT = "open-weight"


class RegisteredModel(BaseModel, frozen=True):
    """A single benchmarked model: identity, display metadata, run config."""

    benchmark: Benchmark
    provider: str
    model: str
    voice: str | None = None  # TTS only
    creator: str | None = None  # who makes the model; None means same as provider
    tags: tuple[ModelTag, ...] = ()
    tenancy: Tenancy = Tenancy.SHARED
    licensing: Licensing = Licensing.PROPRIETARY
    self_hostable: bool = False  # can run in the customer's own infra
    status: ModelStatus
    arena_enabled: bool = True  # in the arena roster? independent of dashboard `status`


_STT = Benchmark.STT
_TTS = Benchmark.TTS
_S2S = Benchmark.S2S
_ACTIVE = ModelStatus.ACTIVE
_RETIRED = ModelStatus.RETIRED
_PENDING = ModelStatus.PENDING
_REALTIME = ModelTag.REALTIME
_MULTI = ModelTag.MULTILINGUAL
_VAD = ModelTag.VAD
_DIAR = ModelTag.DIARIZATION
_TRANS = ModelTag.TRANSLATION
_CODESW = ModelTag.CODE_SWITCHING
_KEYTERM = ModelTag.KEYTERM_BIASING
_CLONE = ModelTag.VOICE_CLONING
_EMOTION = ModelTag.EMOTION_CONTROL
_STREAM = ModelTag.STREAMING_INPUT
_CONVCTX = ModelTag.CONVERSATIONAL_CONTEXT
_OPEN = Licensing.OPEN_WEIGHT

# Per-benchmark order is the model order /v1/providers returns.
MODEL_REGISTRY: list[RegisteredModel] = [
    #######
    # STT #
    #######
    RegisteredModel(
        benchmark=_STT,
        provider="deepgram",
        model="nova-2",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM),
        self_hostable=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="deepgram",
        model="nova-3",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM),
        self_hostable=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="deepgram",
        model="flux-general-en",
        tags=(_REALTIME, _VAD, _KEYTERM),
        self_hostable=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="deepgram",
        model="flux-general-multi",
        tags=(_REALTIME, _MULTI, _VAD, _CODESW, _KEYTERM),
        self_hostable=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="elevenlabs",
        model="scribe_v2_realtime",
        tags=(_REALTIME, _MULTI, _VAD, _CODESW, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="openai",
        model="gpt-realtime-whisper",
        tags=(_REALTIME, _MULTI),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="openai",
        model="gpt-4o-transcribe",
        tags=(_REALTIME, _MULTI, _VAD, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="openai",
        model="gpt-4o-mini-transcribe",
        tags=(_REALTIME, _MULTI, _VAD, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-streaming",
        tags=(_REALTIME, _VAD, _DIAR, _KEYTERM),
        self_hostable=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-streaming-multilingual",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM),
        self_hostable=True,
        status=_ACTIVE,
    ),
    # No longer offered on AssemblyAI's streaming API (u3-rt-pro); superseded by
    # universal-3.5-pro.
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-3-pro",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM),
        self_hostable=True,
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-3.5-pro",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM, _CONVCTX),
        self_hostable=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="speechmatics",
        model="default",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        self_hostable=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="speechmatics",
        model="enhanced",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        self_hostable=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="gradium",
        model="default",
        tags=(_REALTIME, _MULTI, _VAD),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="gladia",
        model="solaria-1",
        tags=(_REALTIME, _MULTI, _VAD, _TRANS, _CODESW, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="soniox",
        model="stt-rt-v4",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="soniox",
        model="stt-rt-v5",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="inworld",
        model="inworld-stt-1",
        tags=(_REALTIME, _MULTI, _VAD, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="xai",
        model="grok-stt",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="smallest",
        model="pulse",
        tags=(_REALTIME, _MULTI, _VAD, _DIAR, _CODESW),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="cartesia",
        model="ink-2",
        tags=(_REALTIME, _VAD),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="mistral",
        model="voxtral-mini-transcribe-realtime-2602",
        tags=(_REALTIME, _MULTI),
        licensing=_OPEN,
        self_hostable=True,
        status=_ACTIVE,
    ),
    # Baseten dedicated endpoints (Whisper Large V3). PENDING: implemented and
    # hidden while Baseten tunes the setup — kept off the scheduled runner and
    # the public site, run manually during the daily test window.
    RegisteredModel(
        benchmark=_STT,
        provider="baseten",
        model="whisper-large-v3",
        creator="openai",
        tags=(_REALTIME, _MULTI, _VAD),
        tenancy=Tenancy.DEDICATED,
        licensing=_OPEN,
        self_hostable=True,
        status=_PENDING,
    ),
    # Azure AI Speech real-time (raw WebSocket, conversation mode).
    RegisteredModel(
        benchmark=_STT,
        provider="azure",
        model="default",
        creator="microsoft",
        tags=(_REALTIME, _MULTI, _VAD),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="google",
        model="chirp_2",
        tags=(_REALTIME, _MULTI, _VAD),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="google",
        model="chirp_3",
        tags=(_REALTIME, _MULTI, _VAD),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="revai",
        model="reverb",
        creator="rev",
        tags=(_REALTIME, _MULTI, _VAD, _KEYTERM),
        licensing=_OPEN,
        self_hostable=True,
        status=_ACTIVE,
    ),
    #######
    # TTS #
    #######
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_flash_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
        tags=(_REALTIME, _MULTI, _CLONE, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        voice="IKne3meq5aSn9XLyUdCD",
        tags=(_REALTIME, _MULTI, _CLONE, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_turbo_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
        tags=(_REALTIME, _MULTI),
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="gpt-4o-mini-tts",
        voice="alloy",
        tags=(_REALTIME, _MULTI, _EMOTION),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="tts-1-hd",
        voice="alloy",
        tags=(_REALTIME, _MULTI),
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="tts-1",
        voice="alloy",
        tags=(_REALTIME, _MULTI),
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic-3",
        voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        tags=(_REALTIME, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic-3.5",
        voice="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        tags=(_REALTIME, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="deepgram",
        model="aura-2-thalia-en",
        voice="aura-2-thalia-en",
        tags=(_REALTIME, _STREAM),
        self_hostable=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="gradium",
        model="default",
        voice="YTpq7expH9539ERJ",
        tags=(_REALTIME, _MULTI, _CLONE, _STREAM),
        status=_ACTIVE,
    ),
    # Rime — all three on /ws3 WebSocket.
    # "arcana" resolves server-side to Arcana v3; "coda" to May 2026 flagship.
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="coda",
        voice="luna",
        tags=(_REALTIME, _MULTI, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="arcana",
        voice="luna",
        tags=(_REALTIME, _MULTI, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="mistv3",
        voice="luna",
        tags=(_REALTIME, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="mistv2",
        voice="luna",
        tags=(_REALTIME, _MULTI),
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="hume",
        model="octave-tts",
        voice="176a55b1-4468-4736-8878-db82729667c1",
        tags=(_REALTIME, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="hume",
        model="octave-2",
        voice="176a55b1-4468-4736-8878-db82729667c1",
        tags=(_REALTIME, _MULTI, _CLONE, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="xai",
        model="grok-tts",
        voice="eve",
        tags=(_REALTIME, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="smallest",
        model="lightning_v3.1_pro",
        voice="kaitlyn",
        tags=(_REALTIME, _MULTI, _CLONE, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="inworld",
        model="inworld-tts-1.5-max",
        voice="Ashley",
        tags=(_REALTIME, _MULTI, _CLONE, _EMOTION, _STREAM),
        self_hostable=True,
        status=_ACTIVE,
        arena_enabled=False,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="inworld",
        model="inworld-tts-1.5-mini",
        voice="Ashley",
        tags=(_REALTIME, _MULTI, _CLONE, _EMOTION, _STREAM),
        self_hostable=True,
        status=_ACTIVE,
        arena_enabled=False,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="soniox",
        model="tts-rt-v1",
        voice="Adrian",
        tags=(_REALTIME, _MULTI, _STREAM),
        status=_ACTIVE,
    ),
    # Azure AI Speech real-time (raw WebSocket). "neural" is the standard
    # neural-voice family; "dragon-hd-latest" pins the auto-updating
    # :DragonHDLatestNeural HD variant. The voice name selects the served model.
    RegisteredModel(
        benchmark=_TTS,
        provider="azure",
        model="neural",
        voice="en-US-AvaNeural",
        creator="microsoft",
        tags=(_REALTIME, _MULTI, _EMOTION, _STREAM),
        status=_ACTIVE,
        arena_enabled=False,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="azure",
        model="dragon-hd-latest",
        voice="en-US-Ava:DragonHDLatestNeural",
        creator="microsoft",
        tags=(_REALTIME, _MULTI, _STREAM),
        status=_ACTIVE,
        arena_enabled=False,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="groq",
        model="canopylabs/orpheus-v1-english",
        voice="autumn",
        creator="canopylabs",
        tags=(_REALTIME, _EMOTION),
        licensing=_OPEN,
        self_hostable=True,
        status=_PENDING,
        arena_enabled=False,
    ),
    # Baseten dedicated endpoints (Qwen3-TTS 1.7B). PENDING for the same reason
    # as the Whisper STT entry above — implemented, hidden, run manually.
    RegisteredModel(
        benchmark=_TTS,
        provider="baseten",
        model="qwen3-tts-1.7b",
        voice="lisa",
        creator="alibaba",
        tags=(_REALTIME, _MULTI, _STREAM),
        tenancy=Tenancy.DEDICATED,
        licensing=_OPEN,
        self_hostable=True,
        status=_PENDING,
    ),
    # gpt-realtime is a speech-to-speech LLM, not a TTS provider: driving it
    # from a text "instructions" prompt folds LLM inference into TTFA and never
    # guarantees verbatim speech, so its metrics are incomparable here. Kept
    # retired (not deleted) so historical rows stay hidden on the site.
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="gpt-realtime-2025-08-28",
        tags=(_REALTIME, _MULTI),
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic",
        tags=(_REALTIME, _MULTI),
        status=_RETIRED,
    ),
    #######
    # S2S #
    #######
    # S2S realtime models. Numbers are fetched daily from Coval (no local
    # provider client). PENDING until the first run lands data.
    RegisteredModel(
        benchmark=_S2S,
        provider="openai",
        model="gpt-realtime",
        tags=(_REALTIME, _MULTI),
        status=_PENDING,
    ),
    RegisteredModel(
        benchmark=_S2S,
        provider="google",
        model="gemini-live",
        tags=(_REALTIME, _MULTI),
        status=_PENDING,
    ),
]

_key_counts = Counter((m.benchmark, m.provider, m.model) for m in MODEL_REGISTRY)
_dupes = sorted(f"{b}:{p}/{m}" for (b, p, m), n in _key_counts.items() if n > 1)
if _dupes:
    raise RuntimeError(f"MODEL_REGISTRY contains duplicate entries: {', '.join(_dupes)}")
