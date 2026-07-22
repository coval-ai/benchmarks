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


class Source(StrEnum):
    """Where the benchmarked endpoint lives: the creator's own API or an inference host."""

    OFFICIAL_API = "official-api"
    SHARED_INFERENCE = "shared-inference"
    DEDICATED_INFERENCE = "dedicated-inference"


class Licensing(StrEnum):
    """Whether the model's weights are openly available."""

    PROPRIETARY = "proprietary"
    OPEN_WEIGHT = "open-weight"


class RegisteredModel(BaseModel, frozen=True, extra="forbid"):
    """A single benchmarked model: identity, display metadata, run config."""

    benchmark: Benchmark
    provider: str
    model: str
    voice: str | None = None  # TTS only
    # TTS only: balanced voice pool, ordered (female, male). When set, each run
    # splits its samples evenly across the pool; ``voice`` is the single-voice
    # fallback for models without one.
    voices: tuple[str, ...] = ()
    creator: str | None = None  # who makes the model; None means same as provider
    tags: tuple[ModelTag, ...] = ()
    source: Source = Source.OFFICIAL_API
    licensing: Licensing = Licensing.PROPRIETARY
    on_prem: bool = False  # provider offers on-prem/customer-infra deployment
    status: ModelStatus
    arena_enabled: bool = True  # in the arena roster? independent of dashboard `status`


_STT = Benchmark.STT
_TTS = Benchmark.TTS
_S2S = Benchmark.S2S
_ACTIVE = ModelStatus.ACTIVE
_RETIRED = ModelStatus.RETIRED
_PENDING = ModelStatus.PENDING
_STREAMING = ModelTag.STREAMING
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
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="deepgram",
        model="nova-3",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="deepgram",
        model="flux-general-en",
        tags=(_STREAMING, _VAD, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="deepgram",
        model="flux-general-multi",
        tags=(_STREAMING, _MULTI, _VAD, _CODESW, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="elevenlabs",
        model="scribe_v2_realtime",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="openai",
        model="gpt-realtime-whisper",
        tags=(_STREAMING, _MULTI),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="openai",
        model="gpt-4o-transcribe",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="openai",
        model="gpt-4o-mini-transcribe",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-streaming",
        tags=(_STREAMING, _VAD, _DIAR, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-streaming-multilingual",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    # No longer offered on AssemblyAI's streaming API (u3-rt-pro); superseded by
    # universal-3.5-pro.
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-3-pro",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM),
        on_prem=True,
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-3.5-pro",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM, _CONVCTX),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="speechmatics",
        model="default",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="speechmatics",
        model="enhanced",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="gradium",
        model="default",
        tags=(_STREAMING, _MULTI, _VAD),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="gladia",
        model="solaria-1",
        tags=(_STREAMING, _MULTI, _VAD, _TRANS, _CODESW, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="soniox",
        model="stt-rt-v4",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="soniox",
        model="stt-rt-v5",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="inworld",
        model="inworld-stt-1",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="xai",
        model="grok-stt",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _KEYTERM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="smallest",
        model="pulse",
        tags=(_STREAMING, _MULTI, _DIAR, _CODESW),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="cartesia",
        model="ink-2",
        tags=(_STREAMING, _VAD),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="mistral",
        model="voxtral-mini-transcribe-realtime-2602",
        tags=(_STREAMING, _MULTI),
        licensing=_OPEN,
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
        tags=(_STREAMING, _MULTI, _VAD),
        source=Source.DEDICATED_INFERENCE,
        licensing=_OPEN,
        on_prem=True,
        status=_PENDING,
    ),
    # Azure AI Speech real-time (raw WebSocket, conversation mode).
    RegisteredModel(
        benchmark=_STT,
        provider="azure",
        model="default",
        creator="microsoft",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="google",
        model="chirp_2",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="google",
        model="chirp_3",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="revai",
        model="reverb",
        creator="rev",
        tags=(_STREAMING, _KEYTERM),
        on_prem=True,
        status=_ACTIVE,
    ),
    # Together AI serverless realtime endpoints (open-weight models).
    RegisteredModel(
        benchmark=_STT,
        provider="together",
        model="nemotron-3-asr-streaming-0.6b",
        creator="nvidia",
        source=Source.SHARED_INFERENCE,
        tags=(_STREAMING,),
        licensing=_OPEN,
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="together",
        model="nemotron-3.5-asr-streaming-0.6b",
        creator="nvidia",
        source=Source.SHARED_INFERENCE,
        tags=(_STREAMING, _MULTI),
        licensing=_OPEN,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="together",
        model="parakeet-tdt-0.6b-v3",
        creator="nvidia",
        source=Source.SHARED_INFERENCE,
        tags=(_STREAMING, _MULTI),
        licensing=_OPEN,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="together",
        model="whisper-large-v3",
        creator="openai",
        source=Source.SHARED_INFERENCE,
        tags=(_STREAMING, _MULTI, _VAD),
        licensing=_OPEN,
        status=_ACTIVE,
    ),
    # Modulate Velma-2 real-time streaming. The empty-frame EOS is a genuine
    # finalize: the complete transcript lands ~150-300 ms after the last audio,
    # so TTFS is comparable and needs no exclusion.
    RegisteredModel(
        benchmark=_STT,
        provider="modulate",
        model="english-fast-transcription-streaming",
        tags=(_STREAMING,),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="modulate",
        model="multilingual-transcription-streaming",
        tags=(_STREAMING, _MULTI, _DIAR),
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
        voices=("21m00Tcm4TlvDq8ikWAM", "29vD33N1CtxCmqQRPOHJ"),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        voice="IKne3meq5aSn9XLyUdCD",
        voices=("21m00Tcm4TlvDq8ikWAM", "29vD33N1CtxCmqQRPOHJ"),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_turbo_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
        tags=(_STREAMING, _MULTI),
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_v3",
        voice="IKne3meq5aSn9XLyUdCD",
        voices=("21m00Tcm4TlvDq8ikWAM", "29vD33N1CtxCmqQRPOHJ"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="gpt-4o-mini-tts",
        voice="alloy",
        voices=("shimmer", "onyx"),
        tags=(_STREAMING, _MULTI, _EMOTION),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="tts-1-hd",
        voice="alloy",
        tags=(_STREAMING, _MULTI),
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="tts-1",
        voice="alloy",
        tags=(_STREAMING, _MULTI),
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic-3",
        voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        on_prem=True,
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic-3.5",
        voice="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        voices=("f786b574-daa5-4673-aa0c-cbe3e8534c02", "a5136bf9-224c-4d76-b823-52bd5efcffcc"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="deepgram",
        model="aura-2-thalia-en",
        voice="aura-2-thalia-en",
        tags=(_STREAMING, _STREAM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="gradium",
        model="default",
        voice="YTpq7expH9539ERJ",
        voices=("NbpkqMVS3CJeq2j8", "6MFfc37kq0sBjBjy"),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="palabra",
        model="palabra-tts-v1",
        voice="default_low",
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        on_prem=True,
        status=_PENDING,
    ),
    # Rime — all three on /ws3 WebSocket.
    # "arcana" resolves server-side to Arcana v3; "coda" to May 2026 flagship.
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="coda",
        voice="luna",
        voices=("luna", "masonry"),
        tags=(_STREAMING, _MULTI, _STREAM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="arcana",
        voice="luna",
        voices=("luna", "masonry"),
        tags=(_STREAMING, _MULTI, _EMOTION, _STREAM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="mistv3",
        voice="luna",
        voices=("luna", "cedar"),
        tags=(_STREAMING, _MULTI, _STREAM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="mistv2",
        voice="luna",
        tags=(_STREAMING, _MULTI),
        on_prem=True,
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="hume",
        model="octave-tts",
        voice="176a55b1-4468-4736-8878-db82729667c1",
        voices=("33045fd9-8010-43f6-b6b0-da3fbf326c29", "82a76fb8-3524-4e87-9265-9795c8e4ede6"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="hume",
        model="octave-2",
        voice="176a55b1-4468-4736-8878-db82729667c1",
        voices=("33045fd9-8010-43f6-b6b0-da3fbf326c29", "82a76fb8-3524-4e87-9265-9795c8e4ede6"),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="xai",
        model="grok-tts",
        voice="eve",
        voices=("eve", "leo"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="smallest",
        model="lightning_v3.1_pro",
        voice="kaitlyn",
        voices=("kaitlyn", "blake"),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="inworld",
        model="inworld-tts-2",
        voice="Ashley",
        voices=("Ashley", "Alex"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="inworld",
        model="inworld-tts-1.5-max",
        voice="Ashley",
        voices=("Ashley", "Alex"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="inworld",
        model="inworld-tts-1.5-mini",
        voice="Ashley",
        voices=("Ashley", "Alex"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="soniox",
        model="tts-rt-v1",
        voice="Adrian",
        voices=("Emma", "Daniel"),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
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
        voices=("en-US-AvaNeural", "en-US-AndrewNeural"),
        creator="microsoft",
        tags=(_STREAMING, _MULTI, _EMOTION, _STREAM),
        on_prem=True,
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="azure",
        model="dragon-hd-latest",
        voice="en-US-Ava:DragonHDLatestNeural",
        voices=("en-US-Ava:DragonHDLatestNeural", "en-US-Andrew:DragonHDLatestNeural"),
        creator="microsoft",
        tags=(_STREAMING, _MULTI, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="groq",
        model="canopylabs/orpheus-v1-english",
        voice="autumn",
        creator="canopylabs",
        source=Source.SHARED_INFERENCE,
        tags=(_STREAMING, _EMOTION),
        status=_PENDING,
        arena_enabled=False,
    ),
    # Google TTS: Gemini buffers input until half-close, hence no streaming-input
    # tag. Arena-disabled: auth is ADC, not a mountable API-key env var.
    RegisteredModel(
        benchmark=_TTS,
        provider="google",
        model="chirp-3-hd",
        voice="en-US-Chirp3-HD-Kore",
        voices=("en-US-Chirp3-HD-Kore", "en-US-Chirp3-HD-Charon"),
        tags=(_STREAMING, _MULTI, _STREAM),
        status=_ACTIVE,
        arena_enabled=False,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="google",
        model="gemini-2.5-flash-tts",
        voice="Kore",
        tags=(_STREAMING, _MULTI, _EMOTION),
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
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        source=Source.DEDICATED_INFERENCE,
        licensing=_OPEN,
        on_prem=True,
        status=_PENDING,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="alibaba",
        model="qwen3-tts-flash-realtime",
        voice="Cherry",
        voices=("Cherry", "Ethan"),
        tags=(_STREAMING, _MULTI, _STREAM),
        status=_ACTIVE,
    ),
    # Fish Audio. Voice is the benchmark-selected speaker used in Fish Audio docs.
    RegisteredModel(
        benchmark=_TTS,
        provider="fishaudio",
        model="s1",
        voice="9a9cf47702da476aa4629e2506d4a857",
        voices=("9a9cf47702da476aa4629e2506d4a857", "536d3a5e000945adb7038665781a4aca"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="fishaudio",
        model="s2.1-pro",
        voice="9a9cf47702da476aa4629e2506d4a857",
        voices=("9a9cf47702da476aa4629e2506d4a857", "536d3a5e000945adb7038665781a4aca"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="fishaudio",
        model="s2.1-pro-free",
        voice="9a9cf47702da476aa4629e2506d4a857",
        voices=("9a9cf47702da476aa4629e2506d4a857", "536d3a5e000945adb7038665781a4aca"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    # MiniMax. Voice is the English narrator MiniMax's own docs use in examples.
    RegisteredModel(
        benchmark=_TTS,
        provider="minimax",
        model="speech-2.8-hd",
        voice="English_expressive_narrator",
        voices=("English_radiant_girl", "English_magnetic_voiced_man"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="minimax",
        model="speech-2.8-turbo",
        voice="English_expressive_narrator",
        voices=("English_radiant_girl", "English_magnetic_voiced_man"),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        status=_ACTIVE,
    ),
    # gpt-realtime is a speech-to-speech LLM, not a TTS provider: driving it
    # from a text "instructions" prompt folds LLM inference into TTFA and never
    # guarantees verbatim speech, so its metrics are incomparable here. Kept
    # retired (not deleted) so historical rows stay hidden on the site.
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="gpt-realtime-2025-08-28",
        tags=(_STREAMING, _MULTI),
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic",
        tags=(_STREAMING, _MULTI),
        status=_RETIRED,
    ),
    #######
    # S2S #
    #######
    # S2S realtime models. Numbers are fetched daily from Coval (no local
    # provider client).
    RegisteredModel(
        benchmark=_S2S,
        provider="openai",
        model="gpt-realtime",
        tags=(_STREAMING, _MULTI),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_S2S,
        provider="google",
        model="gemini-live",
        tags=(_STREAMING, _MULTI),
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_S2S,
        provider="xai",
        model="grok-realtime",
        tags=(_STREAMING, _MULTI),
        # hidden from the site for launch while xAI capacity issues distort
        # latency; the fetch job doesn't read this registry, so data keeps
        # accruing for the flip back to active.
        status=_PENDING,
    ),
]

_key_counts = Counter((m.benchmark, m.provider, m.model) for m in MODEL_REGISTRY)
_dupes = sorted(f"{b}:{p}/{m}" for (b, p, m), n in _key_counts.items() if n > 1)
if _dupes:
    raise RuntimeError(f"MODEL_REGISTRY contains duplicate entries: {', '.join(_dupes)}")
