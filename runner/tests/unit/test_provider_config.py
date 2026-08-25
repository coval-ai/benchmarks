# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the model registry."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

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
from coval_bench.registries.provider_keys import provider_names


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


def test_active_tts_models_have_voices() -> None:
    # The runner can't synthesize without a voice; only non-ACTIVE entries may omit one.
    for m in MODEL_REGISTRY:
        if m.benchmark is Benchmark.TTS and m.status is ModelStatus.ACTIVE:
            assert m.voice is not None, f"{m.provider}/{m.model} is ACTIVE but has no voice"


def test_stt_models_have_no_voice() -> None:
    for m in MODEL_REGISTRY:
        if m.benchmark is Benchmark.STT:
            assert m.voice is None


def test_provider_names_cover_the_class_registries() -> None:
    """Every loadable provider class is reachable by its module name."""
    stt_classes = set(importlib.import_module("coval_bench.providers.stt").STT_PROVIDERS)
    tts_classes = set(importlib.import_module("coval_bench.providers.tts").TTS_PROVIDERS)
    assert stt_classes <= provider_names("stt")
    assert tts_classes <= provider_names("tts")
    # Names with no loadable class must be exactly the optional-SDK providers.
    assert provider_names("stt") - stt_classes <= {"google"}
    assert provider_names("tts") - tts_classes <= {"google", "hume"}


def test_scheduled_models_use_implemented_providers() -> None:
    """Every model the orchestrator may schedule resolves without SDK imports.

    Stopped models are exempt: their rows outlive their provider's code.
    """
    scheduled = (ModelStatus.ACTIVE, ModelStatus.EARLY_ACCESS)
    for model in MODEL_REGISTRY:
        if model.status not in scheduled:
            continue
        if model.benchmark is Benchmark.STT:
            assert model.provider in provider_names("stt"), model.provider
        elif model.benchmark is Benchmark.TTS:
            assert model.provider in provider_names("tts"), model.provider


def test_the_runner_image_installs_every_provider_extra() -> None:
    """The API accepts any provider with a module, so the runner must load them all."""
    runner_root = Path(__file__).parents[2]
    pyproject = tomllib.loads((runner_root / "pyproject.toml").read_text())
    extras = set(pyproject["project"]["optional-dependencies"]) - {"hf-parquet"}  # build-time only
    dockerfile = (runner_root / "Dockerfile").read_text()
    missing = sorted(extra for extra in extras if f"--extra {extra}" not in dockerfile)
    assert not missing, f"provider extras absent from the runner image: {missing}"
