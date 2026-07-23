# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""TTS provider registry.

Usage::

    from coval_bench.providers.tts import TTS_PROVIDERS, HUME_AVAILABLE

    provider_cls = TTS_PROVIDERS["deepgram"]
    provider = provider_cls(settings, model="aura-2-thalia-en", voice="aura-2-thalia-en")
    result = await provider.synthesize("Hello, world!")
"""

from __future__ import annotations

from coval_bench.providers.base import TTSProvider
from coval_bench.providers.tts.alibaba import AlibabaTTSProvider
from coval_bench.providers.tts.azure import AzureTTSProvider
from coval_bench.providers.tts.baseten import BasetenTTSProvider
from coval_bench.providers.tts.cartesia import CartesiaTTSProvider
from coval_bench.providers.tts.deepgram import DeepgramTTSProvider
from coval_bench.providers.tts.elevenlabs import ElevenLabsTTSProvider
from coval_bench.providers.tts.fishaudio import FishAudioTTSProvider
from coval_bench.providers.tts.gradium import GradiumTTSProvider
from coval_bench.providers.tts.groq import GroqTTSProvider
from coval_bench.providers.tts.inworld import InworldTTSProvider
from coval_bench.providers.tts.minimax import MinimaxTTSProvider
from coval_bench.providers.tts.openai import OpenAITTSProvider
from coval_bench.providers.tts.palabra import PalabraTTSProvider
from coval_bench.providers.tts.rime import RimeTTSProvider
from coval_bench.providers.tts.smallest import SmallestTTSProvider
from coval_bench.providers.tts.soniox import SonioxTTSProvider
from coval_bench.providers.tts.speechify import SpeechifyTTSProvider
from coval_bench.providers.tts.xai import XaiTTSProvider

try:
    from coval_bench.providers.tts.hume import HumeTTSProvider

    HUME_AVAILABLE = True
except ImportError:
    HumeTTSProvider = None  # type: ignore[assignment,misc]
    HUME_AVAILABLE = False

try:
    from coval_bench.providers.tts.google import GOOGLE_TTS_AVAILABLE, GoogleTTSProvider
except ImportError:
    GoogleTTSProvider = None  # type: ignore[assignment,misc]
    GOOGLE_TTS_AVAILABLE = False

TTS_PROVIDERS: dict[str, type[TTSProvider]] = {
    "openai": OpenAITTSProvider,
    "cartesia": CartesiaTTSProvider,
    "elevenlabs": ElevenLabsTTSProvider,
    "gradium": GradiumTTSProvider,
    "deepgram": DeepgramTTSProvider,
    "rime": RimeTTSProvider,
    "inworld": InworldTTSProvider,
    "smallest": SmallestTTSProvider,
    "xai": XaiTTSProvider,
    "groq": GroqTTSProvider,
    "soniox": SonioxTTSProvider,
    "baseten": BasetenTTSProvider,
    "azure": AzureTTSProvider,
    "fishaudio": FishAudioTTSProvider,
    "alibaba": AlibabaTTSProvider,
    "minimax": MinimaxTTSProvider,
    "palabra": PalabraTTSProvider,
    "speechify": SpeechifyTTSProvider,
}

if HumeTTSProvider is not None:
    TTS_PROVIDERS["hume"] = HumeTTSProvider

if GoogleTTSProvider is not None and GOOGLE_TTS_AVAILABLE:
    TTS_PROVIDERS["google"] = GoogleTTSProvider

__all__ = [
    "TTS_PROVIDERS",
    "HUME_AVAILABLE",
    "GOOGLE_TTS_AVAILABLE",
    "AlibabaTTSProvider",
    "AzureTTSProvider",
    "BasetenTTSProvider",
    "FishAudioTTSProvider",
    "GradiumTTSProvider",
    "MinimaxTTSProvider",
    "SmallestTTSProvider",
    "XaiTTSProvider",
    "GroqTTSProvider",
    "SonioxTTSProvider",
    "PalabraTTSProvider",
    "SpeechifyTTSProvider",
]
