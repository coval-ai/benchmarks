# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""The models the API test database is seeded with.

The runtime roster lives in ``benchmarks_v2.models``. This seed is the minimum
the API needs to serve something: two live TTS models the arena can pair and
one live STT model. Tests about roster behaviour add their own rows with
``add_models`` in ``tests/api/conftest.py`` or build ``RegisteredModel`` values
inline; nothing here stands for a real provider.
"""

from __future__ import annotations

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import Gender, RegisteredModel, Voice


def _arena_tts(provider: str, model: str) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.TTS,
        provider=provider,
        model=model,
        voice=f"{model}-f",
        voices=(
            Voice(id=f"{model}-f", gender=Gender.FEMALE, name="F", accent="en-US"),
            Voice(id=f"{model}-m", gender=Gender.MALE, name="M", accent="en-US"),
        ),
        collected=True,
        published=True,
    )


TEST_ROSTER: list[RegisteredModel] = [
    RegisteredModel(
        benchmark=Benchmark.STT, provider="seed", model="stt", collected=True, published=True
    ),
    _arena_tts("seed", "tts-a"),
    _arena_tts("seed-b", "tts-b"),
]
