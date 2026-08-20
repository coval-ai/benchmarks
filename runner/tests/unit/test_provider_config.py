# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the model registry."""

from __future__ import annotations

from coval_bench.registries import (
    MODEL_REGISTRY,
    TAG_CATEGORIES,
    Benchmark,
    Licensing,
    ModelTag,
    RegisteredModel,
    Source,
)


def test_every_tag_has_a_category() -> None:
    assert TAG_CATEGORIES.keys() == set(ModelTag)


def test_registry_keys_unique() -> None:
    keys = [(m.benchmark, m.provider, m.model) for m in MODEL_REGISTRY]
    assert len(keys) == len(set(keys))


def test_registered_model_defaults() -> None:
    m = RegisteredModel(benchmark=Benchmark.STT, provider="deepgram", model="nova-3")
    assert m.voice is None
    assert m.creator is None
    assert m.tags == ()
    assert m.source is Source.OFFICIAL_API
    assert m.licensing is Licensing.PROPRIETARY
    assert m.on_prem is False


def test_every_model_has_exactly_one_mode() -> None:
    modes = {ModelTag.STREAMING}
    for m in MODEL_REGISTRY:
        assert len(set(m.tags) & modes) == 1, f"{m.provider}/{m.model} needs one mode tag"


# Legacy entries kept only for their historical rows; both are stopped in
# model_state and must never be flipped running without gaining a voice.
_VOICELESS_TTS = {("openai", "gpt-realtime-2025-08-28"), ("cartesia", "sonic")}


def test_tts_models_have_voices() -> None:
    # The runner can't synthesize without a voice. Run-state lives in the DB,
    # so the registry can't tell which entries are live — every TTS entry
    # outside the pinned legacy set must carry one.
    for m in MODEL_REGISTRY:
        if m.benchmark is Benchmark.TTS and (m.provider, m.model) not in _VOICELESS_TTS:
            assert m.voice is not None, f"{m.provider}/{m.model} has no voice"


def test_stt_models_have_no_voice() -> None:
    for m in MODEL_REGISTRY:
        if m.benchmark is Benchmark.STT:
            assert m.voice is None
