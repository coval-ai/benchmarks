# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Canonical provider → API-key environment variable map.

The single place that ties each TTS provider to the environment variable that
must hold its API key, so provider naming lives in exactly one auditable spot.
Consumed by the cross-repo CI parity check (``scripts/check_arena_keys.py``),
which asserts every arena-eligible provider's key is actually mounted on the
benchmarks-api Cloud Run service in the ``coval-ai/benchmark-infra`` repo.

Names must match the runner's ``Settings`` fields case-insensitively (e.g.
``OPENAI_API_KEY`` ↔ ``settings.openai_api_key``) and the ``name`` of the
secret-backed ``env`` blocks mounted in infra. The ``gradium`` entry is the one
deliberate exception: the TTS client reads ``gradium_tts_api_key`` (there is a
separate ``gradium_api_key`` for the STT provider), so it maps to
``GRADIUM_TTS_API_KEY``.
"""

from __future__ import annotations

import functools
import importlib.util
import pkgutil
from typing import Literal


@functools.cache
def provider_names(kind: Literal["stt", "tts"]) -> frozenset[str]:
    """Provider names for a modality, listed from the provider package's modules.

    A provider module is named exactly its registry key; shared helpers start
    with an underscore. Listing files skips the class imports, so the answer
    does not depend on which optional SDK extras are installed.
    """
    spec = importlib.util.find_spec(f"coval_bench.providers.{kind}")
    if spec is None or spec.submodule_search_locations is None:  # pragma: no cover
        raise RuntimeError(f"provider package coval_bench.providers.{kind} not found")
    return frozenset(
        module.name
        for module in pkgutil.iter_modules(spec.submodule_search_locations)
        if not module.name.startswith("_")
    )


PROVIDER_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "cartesia": "CARTESIA_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
    "gradium": "GRADIUM_TTS_API_KEY",
    "rime": "RIME_API_KEY",
    "hume": "HUME_API_KEY",
    "xai": "XAI_API_KEY",
    "smallest": "SMALLEST_API_KEY",
    "soniox": "SONIOX_API_KEY",
    "inworld": "INWORLD_API_KEY",
    "azure": "AZURE_API_KEY",
    "groq": "GROQ_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "fishaudio": "FISHAUDIO_API_KEY",
    "alibaba": "ALIBABA_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "palabra": "PALABRA_API_KEY",
    "speechify": "SPEECHIFY_API_KEY",
    "lmnt": "LMNT_API_KEY",
    "murf": "MURFAI_API_KEY",
    "hakim": "HAKIMAI_API_KEY",
    "fluxions": "FLUXIONS_API_KEY",
    "deepdub": "DEEPDUB_API_KEY",
}
