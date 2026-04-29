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
from coval_bench.providers.stt.deepgram import DeepgramProvider
from coval_bench.providers.stt.elevenlabs import ElevenLabsSTTProvider
from coval_bench.providers.stt.speechmatics import SpeechmaticsProvider

# Google is optional — gated on the ``google-stt`` extra
try:
    from coval_bench.providers.stt.google import GoogleSTTProvider

    GOOGLE_AVAILABLE: bool = True
except ImportError:
    GoogleSTTProvider = None  # type: ignore[assignment,misc]
    GOOGLE_AVAILABLE = False

STT_PROVIDERS: dict[str, type[STTProvider]] = {
    "deepgram": DeepgramProvider,
    "assemblyai": AssemblyAIProvider,
    "elevenlabs": ElevenLabsSTTProvider,
    "speechmatics": SpeechmaticsProvider,
}

if GoogleSTTProvider is not None:
    STT_PROVIDERS["google"] = GoogleSTTProvider

__all__ = [
    "STT_PROVIDERS",
    "GOOGLE_AVAILABLE",
    "DeepgramProvider",
    "AssemblyAIProvider",
    "ElevenLabsSTTProvider",
    "SpeechmaticsProvider",
    "GoogleSTTProvider",
]
