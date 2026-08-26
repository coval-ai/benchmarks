# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""What one pass of a dataset consumes, read straight from its packaged manifest.

A rate is only half of a cost. The other half is usage, and the manifests
already pin it exactly: the audio an STT dataset feeds a model, and the
characters a TTS dataset asks it to speak. Both are SHA-pinned and identical
for every model, which is what makes a cost comparison between two models fair
— and neither is estimated, so nothing here assumes a speaking rate.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files

from pydantic import BaseModel

from coval_bench.datasets.loader import _load_manifest
from coval_bench.datasets.manifest import STTManifestItem
from coval_bench.registries.benchmarks import Benchmark


class DatasetUsage(BaseModel, frozen=True):
    """One full pass of a dataset, in the units providers bill in."""

    dataset_id: str
    benchmark: Benchmark
    items: int
    # Exactly one is populated: STT datasets are audio, TTS datasets are text.
    audio_minutes: float | None = None
    characters: int | None = None


def _usage_of(dataset_id: str) -> DatasetUsage:
    items = _load_manifest(dataset_id).items
    benchmark = (
        Benchmark.S2S
        if dataset_id.startswith("s2s")
        else Benchmark.STT
        if dataset_id.startswith("stt")
        else Benchmark.TTS
    )
    if items and isinstance(items[0], STTManifestItem):
        seconds = sum(i.duration_sec for i in items if isinstance(i, STTManifestItem))
        return DatasetUsage(
            dataset_id=dataset_id,
            benchmark=benchmark,
            items=len(items),
            audio_minutes=seconds / 60,
        )
    return DatasetUsage(
        dataset_id=dataset_id,
        benchmark=benchmark,
        items=len(items),
        characters=sum(len(i.transcript) for i in items),
    )


@cache
def dataset_usage() -> tuple[DatasetUsage, ...]:
    """Usage for every packaged dataset, by id. Manifests are immutable, so cache."""
    manifests = files("coval_bench.datasets.manifests")
    ids = sorted(
        r.name.removesuffix(".json") for r in manifests.iterdir() if r.name.endswith(".json")
    )
    return tuple(_usage_of(dataset_id) for dataset_id in ids)
