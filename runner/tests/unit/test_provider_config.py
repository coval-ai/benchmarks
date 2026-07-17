# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the model registry."""

from __future__ import annotations

from coval_bench.registries import (
    MODEL_REGISTRY,
    TAG_CATEGORIES,
    Benchmark,
    Licensing,
    ModelStatus,
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
    m = RegisteredModel(
        benchmark=Benchmark.STT, provider="deepgram", model="nova-3", status=ModelStatus.ACTIVE
    )
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


def test_active_tts_models_have_voices() -> None:
    # The runner can't synthesize without a voice; only non-ACTIVE entries may omit one.
    for m in MODEL_REGISTRY:
        if m.benchmark is Benchmark.TTS and m.status is ModelStatus.ACTIVE:
            assert m.voice is not None, f"{m.provider}/{m.model} is ACTIVE but has no voice"


def test_stt_models_have_no_voice() -> None:
    for m in MODEL_REGISTRY:
        if m.benchmark is Benchmark.STT:
            assert m.voice is None
