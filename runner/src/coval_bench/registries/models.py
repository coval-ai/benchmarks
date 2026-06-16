# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry of benchmarked models — identity, run config, and status.

One entry per model, keyed by ``(benchmark, provider, model)``. The
orchestrator runs every ``ACTIVE`` entry; the API serves all entries and
marks ``RETIRED`` ones disabled so the frontend keeps them off the site
even when historical result rows exist for them.
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


class RegisteredModel(BaseModel, frozen=True):
    """A single benchmarked model: identity, display metadata, run config."""

    benchmark: Benchmark
    provider: str
    model: str
    voice: str | None = None  # TTS only
    creator: str | None = None  # who makes the model; None means same as provider
    tags: tuple[ModelTag, ...] = ()
    status: ModelStatus


_STT = Benchmark.STT
_TTS = Benchmark.TTS
_ACTIVE = ModelStatus.ACTIVE
_RETIRED = ModelStatus.RETIRED

# Per-benchmark order is the model order /v1/providers returns.
MODEL_REGISTRY: list[RegisteredModel] = [
    #######
    # STT #
    #######
    RegisteredModel(benchmark=_STT, provider="deepgram", model="nova-2", status=_ACTIVE),
    RegisteredModel(benchmark=_STT, provider="deepgram", model="nova-3", status=_ACTIVE),
    RegisteredModel(benchmark=_STT, provider="deepgram", model="flux-general-en", status=_ACTIVE),
    RegisteredModel(
        benchmark=_STT, provider="deepgram", model="flux-general-multi", status=_ACTIVE
    ),
    RegisteredModel(
        benchmark=_STT, provider="elevenlabs", model="scribe_v2_realtime", status=_ACTIVE
    ),
    RegisteredModel(
        benchmark=_STT, provider="openai", model="gpt-realtime-whisper", status=_ACTIVE
    ),
    RegisteredModel(benchmark=_STT, provider="openai", model="gpt-4o-transcribe", status=_ACTIVE),
    RegisteredModel(
        benchmark=_STT, provider="openai", model="gpt-4o-mini-transcribe", status=_ACTIVE
    ),
    RegisteredModel(
        benchmark=_STT, provider="assemblyai", model="universal-streaming", status=_ACTIVE
    ),
    RegisteredModel(benchmark=_STT, provider="speechmatics", model="default", status=_ACTIVE),
    RegisteredModel(benchmark=_STT, provider="speechmatics", model="enhanced", status=_ACTIVE),
    RegisteredModel(benchmark=_STT, provider="gradium", model="default", status=_ACTIVE),
    RegisteredModel(benchmark=_STT, provider="soniox", model="stt-rt-v4", status=_ACTIVE),
    RegisteredModel(benchmark=_STT, provider="soniox", model="stt-rt-v5", status=_ACTIVE),
    RegisteredModel(benchmark=_STT, provider="xai", model="grok-stt", status=_ACTIVE),
    RegisteredModel(benchmark=_STT, provider="smallest", model="pulse", status=_ACTIVE),
    RegisteredModel(benchmark=_STT, provider="cartesia", model="ink-2", status=_ACTIVE),
    RegisteredModel(benchmark=_STT, provider="google", model="short", status=_RETIRED),
    RegisteredModel(benchmark=_STT, provider="google", model="long", status=_RETIRED),
    RegisteredModel(benchmark=_STT, provider="google", model="telephony", status=_RETIRED),
    RegisteredModel(benchmark=_STT, provider="google", model="chirp_2", status=_RETIRED),
    #######
    # TTS #
    #######
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_flash_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        voice="IKne3meq5aSn9XLyUdCD",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_turbo_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS, provider="openai", model="gpt-4o-mini-tts", voice="alloy", status=_ACTIVE
    ),
    RegisteredModel(
        benchmark=_TTS, provider="openai", model="tts-1-hd", voice="alloy", status=_RETIRED
    ),
    RegisteredModel(
        benchmark=_TTS, provider="openai", model="tts-1", voice="alloy", status=_RETIRED
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic-3",
        voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic-3.5",
        voice="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="deepgram",
        model="aura-2-thalia-en",
        voice="aura-2-thalia-en",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="gradium",
        model="default",
        voice="YTpq7expH9539ERJ",
        status=_ACTIVE,
    ),
    # Rime — all three on /ws3 WebSocket.
    # "arcana" resolves server-side to Arcana v3; "coda" to May 2026 flagship.
    RegisteredModel(benchmark=_TTS, provider="rime", model="coda", voice="luna", status=_ACTIVE),
    RegisteredModel(benchmark=_TTS, provider="rime", model="arcana", voice="luna", status=_ACTIVE),
    RegisteredModel(benchmark=_TTS, provider="rime", model="mistv3", voice="luna", status=_ACTIVE),
    RegisteredModel(benchmark=_TTS, provider="rime", model="mistv2", voice="luna", status=_RETIRED),
    RegisteredModel(
        benchmark=_TTS,
        provider="hume",
        model="octave-tts",
        voice="176a55b1-4468-4736-8878-db82729667c1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="hume",
        model="octave-2",
        voice="176a55b1-4468-4736-8878-db82729667c1",
        status=_ACTIVE,
    ),
    RegisteredModel(benchmark=_TTS, provider="xai", model="grok-tts", voice="eve", status=_ACTIVE),
    RegisteredModel(
        benchmark=_TTS,
        provider="smallest",
        model="lightning_v3.1_pro",
        voice="kaitlyn",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS, provider="soniox", model="tts-rt-v1", voice="Adrian", status=_ACTIVE
    ),
    # gpt-realtime is a speech-to-speech LLM, not a TTS provider: driving it
    # from a text "instructions" prompt folds LLM inference into TTFA and never
    # guarantees verbatim speech, so its metrics are incomparable here. Kept
    # retired (not deleted) so historical rows stay hidden on the site.
    RegisteredModel(
        benchmark=_TTS, provider="openai", model="gpt-realtime-2025-08-28", status=_RETIRED
    ),
    RegisteredModel(benchmark=_TTS, provider="cartesia", model="sonic", status=_RETIRED),
]

_key_counts = Counter((m.benchmark, m.provider, m.model) for m in MODEL_REGISTRY)
_dupes = sorted(f"{b}:{p}/{m}" for (b, p, m), n in _key_counts.items() if n > 1)
if _dupes:
    raise RuntimeError(f"MODEL_REGISTRY contains duplicate entries: {', '.join(_dupes)}")
