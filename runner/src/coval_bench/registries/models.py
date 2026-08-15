# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry of benchmarked models — identity, run config, and status.

One entry per model, keyed by ``(benchmark, provider, model)``. The
orchestrator runs every ``ACTIVE`` and ``EARLY_ACCESS`` entry; the API omits
``EARLY_ACCESS`` entries from public responses and marks ``RETIRED``/``PENDING``
ones disabled so the frontend keeps them off the site even when historical
result rows exist for them.
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
    # Pre-launch embargo: run normally, served only to internal-key requests.
    EARLY_ACCESS = "early-access"


class Source(StrEnum):
    """Where the benchmarked endpoint lives: the creator's own API or an inference host."""

    OFFICIAL_API = "official-api"
    SHARED_INFERENCE = "shared-inference"
    DEDICATED_INFERENCE = "dedicated-inference"


class Licensing(StrEnum):
    """Whether the model's weights are openly available."""

    PROPRIETARY = "proprietary"
    OPEN_WEIGHT = "open-weight"


class Gender(StrEnum):
    """Perceived gender of a voice — an editorial label, not ground truth.

    Provider metadata is too uneven to encode provenance: Hume tags every stock
    voice, OpenAI publishes none. Four providers here are API-confirmed
    (cartesia, hume, xai, lmnt); the rest are our own listening.
    """

    FEMALE = "female"
    MALE = "male"


class Voice(BaseModel, frozen=True, extra="forbid"):
    """One speaker, as the provider addresses it.

    ``id`` goes on the wire and is often an opaque UUID, so ``name`` keeps
    registry diffs readable. ``accent`` is recorded where known, not controlled.

    Nothing pairs on ``gender`` yet: the arena synthesizes with the scalar
    ``RegisteredModel.voice``, which has no gender and for most models isn't
    even a pool member. Battles today are cross-gender by construction.
    """

    id: str
    gender: Gender
    name: str | None = None
    accent: str | None = None


class RegisteredModel(BaseModel, frozen=True, extra="forbid"):
    """A single benchmarked model: identity, display metadata, run config."""

    benchmark: Benchmark
    provider: str
    model: str
    voice: str | None = None  # TTS only
    # TTS only: balanced voice pool, one :class:`Voice` per gender. Each run
    # splits its samples evenly across the pool; ``voice`` is the fallback for
    # models without one. Gender lives on the ``Voice``, not in tuple order —
    # see :class:`Voice` for why nothing pairs on it yet.
    voices: tuple[Voice, ...] = ()
    creator: str | None = None  # who makes the model; None means same as provider
    tags: tuple[ModelTag, ...] = ()
    source: Source = Source.OFFICIAL_API
    licensing: Licensing = Licensing.PROPRIETARY
    on_prem: bool = False  # provider offers on-prem/customer-infra deployment
    # Serving region behind the endpoint we benchmark, normalized to
    # ``us-east-1``-style codes regardless of cloud. Broad ``us``/``eu`` when
    # only the geography is known; ``global`` when the provider documents
    # globally routed serving. Unset means unknown. Geo-routed endpoints
    # record where our runner's traffic lands.
    region: str | None = None
    status: ModelStatus
    arena_enabled: bool = True  # in the arena roster? independent of dashboard `status`


_STT = Benchmark.STT
_TTS = Benchmark.TTS
_S2S = Benchmark.S2S
_ACTIVE = ModelStatus.ACTIVE
_PAUSED = ModelStatus.PAUSED
_RETIRED = ModelStatus.RETIRED
_PENDING = ModelStatus.PENDING
_EARLY_ACCESS = ModelStatus.EARLY_ACCESS
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
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="deepgram",
        model="nova-3",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM),
        on_prem=True,
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="deepgram",
        model="flux-general-en",
        tags=(_STREAMING, _VAD, _KEYTERM),
        on_prem=True,
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="deepgram",
        model="flux-general-multi",
        tags=(_STREAMING, _MULTI, _VAD, _CODESW, _KEYTERM),
        on_prem=True,
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="elevenlabs",
        model="scribe_v2_realtime",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="openai",
        model="gpt-realtime-whisper",
        tags=(_STREAMING, _MULTI),
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="openai",
        model="gpt-4o-transcribe",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="openai",
        model="gpt-4o-mini-transcribe",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-streaming",
        tags=(_STREAMING, _VAD, _DIAR, _KEYTERM),
        on_prem=True,
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-streaming-multilingual",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM),
        on_prem=True,
        region="us-east-1",
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
        region="us-east-1",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="assemblyai",
        model="universal-3.5-pro",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _CODESW, _KEYTERM, _CONVCTX),
        on_prem=True,
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="speechmatics",
        model="default",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        on_prem=True,
        region="eu-west-3",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="speechmatics",
        model="enhanced",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        on_prem=True,
        region="eu-west-3",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="gradium",
        model="default",
        tags=(_STREAMING, _MULTI, _VAD),
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="gladia",
        model="solaria-1",
        tags=(_STREAMING, _MULTI, _VAD, _TRANS, _CODESW, _KEYTERM),
        region="eu-west",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="soniox",
        model="stt-rt-v4",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="soniox",
        model="stt-rt-v5",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _TRANS, _CODESW, _KEYTERM),
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="inworld",
        model="inworld-stt-1",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="xai",
        model="grok-stt",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _KEYTERM),
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="smallest",
        model="pulse",
        tags=(_STREAMING, _MULTI, _DIAR, _CODESW),
        region="us-west-2",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="cartesia",
        model="ink-2",
        tags=(_STREAMING, _VAD),
        on_prem=True,
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="mistral",
        model="voxtral-mini-transcribe-realtime-2602",
        tags=(_STREAMING, _MULTI),
        licensing=_OPEN,
        region="eu",
        status=_ACTIVE,
    ),
    # Baseten dedicated endpoint (Whisper Large V3). Benchmarked nightly by the
    # dedicated runner job during Baseten's test window; EARLY_ACCESS keeps it
    # off the public site until the dedicated-inference dashboards ship.
    RegisteredModel(
        benchmark=_STT,
        provider="baseten",
        model="whisper-large-v3",
        creator="openai",
        tags=(_STREAMING, _MULTI, _VAD),
        source=Source.DEDICATED_INFERENCE,
        licensing=_OPEN,
        on_prem=True,
        status=_EARLY_ACCESS,
    ),
    # Azure AI Speech real-time (raw WebSocket, conversation mode).
    RegisteredModel(
        benchmark=_STT,
        provider="azure",
        model="default",
        creator="microsoft",
        tags=(_STREAMING, _MULTI, _VAD, _DIAR, _KEYTERM),
        on_prem=True,
        region="us-east-1",
        status=_PAUSED,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="google",
        model="chirp_2",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        on_prem=True,
        region="us-central-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="google",
        model="chirp_3",
        tags=(_STREAMING, _MULTI, _VAD, _KEYTERM),
        on_prem=True,
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="revai",
        model="reverb",
        creator="rev",
        tags=(_STREAMING, _KEYTERM),
        on_prem=True,
        region="us-west-2",
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
        region="us",
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
        region="us",
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
        region="us",
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
        region="us",
        status=_ACTIVE,
    ),
    # Reson8's realtime endpoint takes no model id, so the slug names the endpoint
    # and leaves room for their separate turn-level one. TTFT is grid-quantized
    # and excluded in registries/metrics.py; the client-driven flush keeps TTFS
    # comparable.
    RegisteredModel(
        benchmark=_STT,
        provider="reson8",
        model="realtime",
        tags=(_STREAMING, _MULTI, _DIAR, _KEYTERM),
        region="eu",
        status=_ACTIVE,
    ),
    # Modulate Velma-2 real-time streaming; ids are the endpoint path segments.
    # The empty-frame EOS is a genuine finalize: the complete transcript lands
    # ~150-300 ms after the last audio, so TTFS is comparable. The English
    # endpoint's TTFT is cadence-floored and excluded in registries/metrics.py.
    RegisteredModel(
        benchmark=_STT,
        provider="modulate",
        model="velma-2-stt-streaming-english-v2",
        tags=(_STREAMING,),
        region="us-east-1",
        status=_EARLY_ACCESS,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="modulate",
        model="velma-2-stt-streaming",
        tags=(_STREAMING, _MULTI, _DIAR),
        region="us-east-1",
        status=_ACTIVE,
    ),
    # Pre-Velma-2 ids; retired so their orphaned result rows stay off the site.
    RegisteredModel(
        benchmark=_STT,
        provider="modulate",
        model="english-fast-transcription-streaming",
        tags=(_STREAMING,),
        region="us-east-1",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_STT,
        provider="modulate",
        model="multilingual-transcription-streaming",
        tags=(_STREAMING, _MULTI, _DIAR),
        region="us-east-1",
        status=_RETIRED,
    ),
    #######
    # TTS #
    #######
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_flash_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        voice="IKne3meq5aSn9XLyUdCD",
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_turbo_v2_5",
        voice="IKne3meq5aSn9XLyUdCD",
        tags=(_STREAMING, _MULTI),
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_v3",
        voice="IKne3meq5aSn9XLyUdCD",
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION),
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="elevenlabs",
        model="eleven_v3_conversational",
        voice="IKne3meq5aSn9XLyUdCD",
        voices=(
            Voice(id="21m00Tcm4TlvDq8ikWAM", gender=Gender.FEMALE, name="Rachel"),
            Voice(id="29vD33N1CtxCmqQRPOHJ", gender=Gender.MALE, name="Drew"),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us",
        status=_EARLY_ACCESS,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="gpt-4o-mini-tts",
        voice="alloy",
        voices=(Voice(id="shimmer", gender=Gender.FEMALE), Voice(id="onyx", gender=Gender.MALE)),
        tags=(_STREAMING, _MULTI, _EMOTION),
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="tts-1-hd",
        voice="alloy",
        tags=(_STREAMING, _MULTI),
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="openai",
        model="tts-1",
        voice="alloy",
        tags=(_STREAMING, _MULTI),
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic-3",
        voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        on_prem=True,
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic-3.5",
        voice="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        voices=(
            Voice(
                id="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
                gender=Gender.FEMALE,
                name="Skylar",
                accent="en-US",
            ),
            Voice(
                id="30894953-bcce-41fe-892c-15ce19c843ff",
                gender=Gender.MALE,
                name="Parker",
                accent="en-US",
            ),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        on_prem=True,
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="deepgram",
        model="aura-2-thalia-en",
        voice="aura-2-thalia-en",
        voices=(
            Voice(id="aura-2-thalia-en", gender=Gender.FEMALE, name="Thalia", accent="en-US"),
            Voice(id="aura-2-orion-en", gender=Gender.MALE, name="Orion", accent="en-US"),
        ),
        tags=(_STREAMING, _STREAM),
        on_prem=True,
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="deepgram",
        model="flux-haley-en",
        voice="flux-haley-en",
        voices=(
            Voice(id="flux-haley-en", gender=Gender.FEMALE, name="Haley", accent="en-US"),
            Voice(id="flux-cole-en", gender=Gender.MALE, name="Cole", accent="en-US"),
        ),
        tags=(_STREAMING, _STREAM),
        on_prem=True,
        region="us",
        status=_EARLY_ACCESS,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="gradium",
        model="default",
        voice="YTpq7expH9539ERJ",
        # Genders are positional carry-over, NOT verified: Gradium's catalog isn't
        # enumerable and `voice` above appears in neither pool slot.
        voices=(
            Voice(id="NbpkqMVS3CJeq2j8", gender=Gender.FEMALE),
            Voice(id="6MFfc37kq0sBjBjy", gender=Gender.MALE),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="palabra",
        model="palabra-tts-v1",
        voice="default_low",
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        on_prem=True,
        region="us-central-1",
        status=_ACTIVE,
    ),
    # Rime — all three on /ws3 WebSocket.
    # "arcana" resolves server-side to Arcana v3; "coda" to May 2026 flagship.
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="coda",
        voice="luna",
        voices=(Voice(id="luna", gender=Gender.FEMALE), Voice(id="masonry", gender=Gender.MALE)),
        tags=(_STREAMING, _MULTI, _STREAM),
        on_prem=True,
        region="us-west-2",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="arcana",
        voice="luna",
        tags=(_STREAMING, _MULTI, _EMOTION, _STREAM),
        on_prem=True,
        region="us-west-2",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="mistv3",
        voice="luna",
        voices=(Voice(id="luna", gender=Gender.FEMALE), Voice(id="cedar", gender=Gender.MALE)),
        tags=(_STREAMING, _MULTI, _STREAM),
        on_prem=True,
        region="us-west-2",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="rime",
        model="mistv2",
        voice="luna",
        tags=(_STREAMING, _MULTI),
        on_prem=True,
        region="us-west-2",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="hume",
        model="octave-tts",
        voice="176a55b1-4468-4736-8878-db82729667c1",
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="hume",
        model="octave-2",
        voice="176a55b1-4468-4736-8878-db82729667c1",
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="xai",
        model="grok-tts",
        voice="carina",
        voices=(Voice(id="carina", gender=Gender.FEMALE), Voice(id="altair", gender=Gender.MALE)),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="smallest",
        model="lightning_v3.1_pro",
        voice="kelsey",
        voices=(Voice(id="kelsey", gender=Gender.FEMALE), Voice(id="spencer", gender=Gender.MALE)),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        region="us-west-2",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="inworld",
        model="inworld-tts-2",
        voice="Brooke",
        voices=(
            Voice(id="Brooke", gender=Gender.FEMALE, accent="en-US"),
            Voice(id="Jason", gender=Gender.MALE, accent="en-US"),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="inworld",
        model="inworld-tts-2-flash",
        voice="Brooke",
        voices=(
            Voice(id="Brooke", gender=Gender.FEMALE, accent="en-US"),
            Voice(id="Jason", gender=Gender.MALE, accent="en-US"),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="inworld",
        model="inworld-tts-1.5-max",
        voice="Brooke",
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        on_prem=True,
        region="us-east-1",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="inworld",
        model="inworld-tts-1.5-mini",
        voice="Brooke",
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        on_prem=True,
        region="us-east-1",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="soniox",
        model="tts-rt-v1",
        voice="Adrian",
        voices=(Voice(id="Emma", gender=Gender.FEMALE), Voice(id="Daniel", gender=Gender.MALE)),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        region="us",
        status=_ACTIVE,
        arena_enabled=False,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="soniox",
        model="tts-rt-v2",
        voice="Adrian",
        voices=(Voice(id="Emma", gender=Gender.FEMALE), Voice(id="Daniel", gender=Gender.MALE)),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        region="us",
        status=_ACTIVE,
        arena_enabled=False,
    ),
    # Azure AI Speech real-time (raw WebSocket). "neural" is the standard
    # neural-voice family; "dragon-hd-latest" pins the auto-updating
    # :DragonHDLatestNeural HD variant. The voice name selects the served model.
    RegisteredModel(
        benchmark=_TTS,
        provider="azure",
        model="neural",
        voice="en-US-AvaNeural",
        voices=(
            Voice(id="en-US-AvaNeural", gender=Gender.FEMALE, name="Ava", accent="en-US"),
            Voice(id="en-US-AndrewNeural", gender=Gender.MALE, name="Andrew", accent="en-US"),
        ),
        creator="microsoft",
        tags=(_STREAMING, _MULTI, _EMOTION, _STREAM),
        on_prem=True,
        region="us-east-1",
        status=_PAUSED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="azure",
        model="dragon-hd-latest",
        voice="en-US-Ava:DragonHDLatestNeural",
        voices=(
            Voice(
                id="en-US-Ava:DragonHDLatestNeural",
                gender=Gender.FEMALE,
                name="Ava",
                accent="en-US",
            ),
            Voice(
                id="en-US-Andrew:DragonHDLatestNeural",
                gender=Gender.MALE,
                name="Andrew",
                accent="en-US",
            ),
        ),
        creator="microsoft",
        tags=(_STREAMING, _MULTI, _STREAM),
        region="us-east-1",
        status=_PAUSED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="groq",
        model="canopylabs/orpheus-v1-english",
        voice="autumn",
        creator="canopylabs",
        source=Source.SHARED_INFERENCE,
        tags=(_STREAMING, _EMOTION),
        region="us",
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
        voices=(
            Voice(id="en-US-Chirp3-HD-Kore", gender=Gender.FEMALE, name="Kore", accent="en-US"),
            Voice(
                id="en-US-Chirp3-HD-Charon",
                gender=Gender.MALE,
                name="Charon",
                accent="en-US",
            ),
        ),
        tags=(_STREAMING, _MULTI, _STREAM),
        region="global",
        status=_ACTIVE,
        arena_enabled=False,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="google",
        model="gemini-2.5-flash-tts",
        voice="Kore",
        tags=(_STREAMING, _MULTI, _EMOTION),
        region="global",
        status=_PENDING,
        arena_enabled=False,
    ),
    # Baseten dedicated endpoint (Qwen3-TTS 1.7B). Same treatment as the
    # Whisper STT entry above — nightly dedicated runner job, EARLY_ACCESS.
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
        status=_EARLY_ACCESS,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="alibaba",
        model="qwen3-tts-flash-realtime",
        voice="Cherry",
        voices=(Voice(id="Cherry", gender=Gender.FEMALE), Voice(id="Ethan", gender=Gender.MALE)),
        tags=(_STREAMING, _MULTI, _STREAM),
        region="ap-southeast-1",
        status=_ACTIVE,
    ),
    # Fish Audio. Voice ids are library reference_ids, not names.
    RegisteredModel(
        benchmark=_TTS,
        provider="fishaudio",
        model="s1",
        voice="4501d82f5de3467ebf4d7ef095a2deee",
        voices=(
            Voice(id="4501d82f5de3467ebf4d7ef095a2deee", gender=Gender.FEMALE, name="Marlowe"),
            Voice(id="fa4c9eb3dccc4806b382b40d61c6b10a", gender=Gender.MALE, name="Sawyer"),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us-west-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="fishaudio",
        model="s2.1-pro",
        voice="4501d82f5de3467ebf4d7ef095a2deee",
        voices=(
            Voice(id="4501d82f5de3467ebf4d7ef095a2deee", gender=Gender.FEMALE, name="Marlowe"),
            Voice(id="fa4c9eb3dccc4806b382b40d61c6b10a", gender=Gender.MALE, name="Sawyer"),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us-west-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="fishaudio",
        model="s2.1-pro-free",
        voice="4501d82f5de3467ebf4d7ef095a2deee",
        voices=(
            Voice(id="4501d82f5de3467ebf4d7ef095a2deee", gender=Gender.FEMALE, name="Marlowe"),
            Voice(id="fa4c9eb3dccc4806b382b40d61c6b10a", gender=Gender.MALE, name="Sawyer"),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us-west-1",
        status=_ACTIVE,
    ),
    # MiniMax. Voice is the English narrator MiniMax's own docs use in examples.
    RegisteredModel(
        benchmark=_TTS,
        provider="minimax",
        model="speech-2.8-hd",
        voice="English_expressive_narrator",
        voices=(
            Voice(id="English_radiant_girl", gender=Gender.FEMALE),
            Voice(id="English_magnetic_voiced_man", gender=Gender.MALE),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="minimax",
        model="speech-2.8-turbo",
        voice="English_expressive_narrator",
        voices=(
            Voice(id="English_radiant_girl", gender=Gender.FEMALE),
            Voice(id="English_magnetic_voiced_man", gender=Gender.MALE),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us-east-1",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="speechify",
        model="simba-3.2",
        voice="geffen_32",
        voices=(
            Voice(id="beatrice_32", gender=Gender.FEMALE),
            Voice(id="hugh_32", gender=Gender.MALE),
        ),
        tags=(_STREAMING, _CLONE, _EMOTION),
        region="us-east",
        status=_ACTIVE,
        arena_enabled=False,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="speechify",
        model="simba-3.0",
        voice="geffen_32",
        voices=(
            Voice(id="beatrice_32", gender=Gender.FEMALE),
            Voice(id="hugh_32", gender=Gender.MALE),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION),
        region="us-east",
        status=_ACTIVE,
        arena_enabled=False,
    ),
    # No model id on the wire, only a voice, so "vui" is the bare surface name.
    RegisteredModel(
        benchmark=_TTS,
        provider="fluxions",
        model="vui",
        voice="delta",
        voices=(
            Voice(id="delta", gender=Gender.FEMALE, accent="en-US"),
            Voice(id="grove", gender=Gender.MALE, accent="en-US"),
        ),
        tags=(_STREAMING, _CLONE, _EMOTION),
        status=_ACTIVE,
        arena_enabled=False,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="lmnt",
        model="blizzard",
        voice="leah",
        voices=(Voice(id="leah", gender=Gender.FEMALE), Voice(id="caleb", gender=Gender.MALE)),
        tags=(_STREAMING, _MULTI, _CLONE, _STREAM),
        region="us",
        status=_ACTIVE,
    ),
    # Deepdub eTTS. Voices are preset voice-prompt ids from Deepdub's docs:
    # the female sales-agent and male call-center-agent presets.
    RegisteredModel(
        benchmark=_TTS,
        provider="deepdub",
        model="dd-etts-3.3",
        voice="02215cf5-04af-46f3-a061-48a4c81989bf",
        voices=(
            Voice(id="02215cf5-04af-46f3-a061-48a4c81989bf", gender=Gender.FEMALE),
            Voice(id="b2abb241-ac92-48bf-a890-fec03f43e209", gender=Gender.MALE),
        ),
        tags=(_STREAMING, _MULTI, _CLONE, _EMOTION, _STREAM),
        region="us-east-1",
        status=_EARLY_ACCESS,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="murf",
        model="falcon-2",
        voice="Amara",
        voices=(Voice(id="Amara", gender=Gender.FEMALE), Voice(id="Gordon", gender=Gender.MALE)),
        tags=(_STREAMING, _MULTI, _STREAM),
        region="us-east-2",
        status=_ACTIVE,
        arena_enabled=False,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="hakim",
        model="hakim-fast-v1",
        voice="amelia-en-us",
        voices=(
            Voice(id="amelia-en-us", gender=Gender.FEMALE, name="Amelia", accent="en-US"),
            Voice(id="noah-en-us", gender=Gender.MALE, name="Noah", accent="en-US"),
        ),
        tags=(_STREAMING, _MULTI, _CLONE),
        region="eu-central-1",
        status=_EARLY_ACCESS,
        arena_enabled=False,
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
        region="us",
        status=_RETIRED,
    ),
    RegisteredModel(
        benchmark=_TTS,
        provider="cartesia",
        model="sonic",
        tags=(_STREAMING, _MULTI),
        region="us",
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
        region="us",
        status=_ACTIVE,
    ),
    RegisteredModel(
        benchmark=_S2S,
        provider="google",
        model="gemini-live",
        tags=(_STREAMING, _MULTI),
        region="global",
        status=_ACTIVE,
    ),
    # xAI stays under the early-access embargo while they are unresponsive to
    # outreach: runs and fetches normally, but every data endpoint strips both
    # models for public callers (unlike PENDING, which only disables the
    # catalogue entry and still serves their rows).
    RegisteredModel(
        benchmark=_S2S,
        provider="xai",
        model="grok-voice-think-fast-1.0",
        tags=(_STREAMING, _MULTI),
        region="us-east-1",
        status=_EARLY_ACCESS,
    ),
    RegisteredModel(
        benchmark=_S2S,
        provider="xai",
        model="grok-voice-think-fast-2.0",
        tags=(_STREAMING, _MULTI),
        region="us-east-1",
        status=_EARLY_ACCESS,
    ),
    # Pre-launch models named by codename only: these strings reach the results
    # table and every surface built on it, so they carry no vendor identity of
    # their own. Leave `creator` unset — it falls back to the provider, which is
    # the point.
    RegisteredModel(
        benchmark=_S2S,
        provider="colors",
        model="gray",
        tags=(_STREAMING, _MULTI),
        status=_EARLY_ACCESS,
    ),
    RegisteredModel(
        benchmark=_S2S,
        provider="colors",
        model="red",
        tags=(_STREAMING, _MULTI),
        status=_EARLY_ACCESS,
    ),
]

_key_counts = Counter((m.benchmark, m.provider, m.model) for m in MODEL_REGISTRY)
_dupes = sorted(f"{b}:{p}/{m}" for (b, p, m), n in _key_counts.items() if n > 1)
if _dupes:
    raise RuntimeError(f"MODEL_REGISTRY contains duplicate entries: {', '.join(_dupes)}")
