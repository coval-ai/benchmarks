# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""STT provider registry.

Usage::

    from coval_bench.providers.stt import STT_PROVIDERS

    for name, cls in STT_PROVIDERS.items():
        provider = cls(api_key=settings.deepgram_api_key, model="nova-3")
        result = await provider.measure_ttft(...)
"""

from __future__ import annotations

from coval_bench.providers.base import STTProvider
from coval_bench.providers.stt.assemblyai import AssemblyAIProvider
from coval_bench.providers.stt.baseten import BasetenSTTProvider
from coval_bench.providers.stt.cartesia import CartesiaSTTProvider
from coval_bench.providers.stt.deepgram import DeepgramProvider
from coval_bench.providers.stt.elevenlabs import ElevenLabsSTTProvider
from coval_bench.providers.stt.gladia import GladiaSTTProvider
from coval_bench.providers.stt.gradium import GradiumSTTProvider
from coval_bench.providers.stt.inworld import InworldSTTProvider
from coval_bench.providers.stt.openai import OpenAISTTProvider
from coval_bench.providers.stt.smallest import SmallestSTTProvider
from coval_bench.providers.stt.soniox import SonioxSTTProvider
from coval_bench.providers.stt.speechmatics import SpeechmaticsProvider
from coval_bench.providers.stt.xai import XaiSTTProvider

# Google is optional — gated on the ``google-stt`` extra
try:
    from coval_bench.providers.stt.google import GoogleSTTProvider

    GOOGLE_AVAILABLE: bool = True
except ImportError:
    GoogleSTTProvider = None  # type: ignore[assignment,misc]
    GOOGLE_AVAILABLE = False

STT_PROVIDERS: dict[str, type[STTProvider]] = {
    "deepgram": DeepgramProvider,
    "cartesia": CartesiaSTTProvider,
    "assemblyai": AssemblyAIProvider,
    "baseten": BasetenSTTProvider,
    "elevenlabs": ElevenLabsSTTProvider,
    "gladia": GladiaSTTProvider,
    "gradium": GradiumSTTProvider,
    "inworld": InworldSTTProvider,
    "openai": OpenAISTTProvider,
    "smallest": SmallestSTTProvider,
    "soniox": SonioxSTTProvider,
    "speechmatics": SpeechmaticsProvider,
    "xai": XaiSTTProvider,
}

if GoogleSTTProvider is not None:
    STT_PROVIDERS["google"] = GoogleSTTProvider

__all__ = [
    "STT_PROVIDERS",
    "GOOGLE_AVAILABLE",
    "DeepgramProvider",
    "CartesiaSTTProvider",
    "AssemblyAIProvider",
    "BasetenSTTProvider",
    "ElevenLabsSTTProvider",
    "GladiaSTTProvider",
    "GradiumSTTProvider",
    "InworldSTTProvider",
    "OpenAISTTProvider",
    "SmallestSTTProvider",
    "SonioxSTTProvider",
    "SpeechmaticsProvider",
    "XaiSTTProvider",
    "GoogleSTTProvider",
]
