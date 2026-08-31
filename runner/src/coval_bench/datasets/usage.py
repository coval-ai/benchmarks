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
    # The id prefix names the benchmark, and the benchmark decides the billing
    # axis. Both raise on a dataset that breaks the convention — a wrong-axis
    # or wrong-benchmark row in a public payload must never ship silently.
    benchmark = Benchmark(dataset_id.split("-", 1)[0].upper())
    if benchmark is Benchmark.TTS:
        return DatasetUsage(
            dataset_id=dataset_id,
            benchmark=benchmark,
            items=len(items),
            characters=sum(len(i.transcript) for i in items),
        )
    audio_items = [i for i in items if isinstance(i, STTManifestItem)]
    if len(audio_items) != len(items):
        raise ValueError(f"{dataset_id} bills audio in, but its manifest items carry no duration")
    return DatasetUsage(
        dataset_id=dataset_id,
        benchmark=benchmark,
        items=len(items),
        audio_minutes=sum(i.duration_sec for i in audio_items) / 60,
    )


@cache
def dataset_usage() -> tuple[DatasetUsage, ...]:
    """Usage for every packaged dataset, by id. Manifests are immutable, so cache.

    Every packaged manifest is served, including datasets no longer in the run
    rotation — the package is the only registry of datasets this repo has, and
    their historical artifacts still reference them. A consumer costing a
    current benchmark pass should multiply against the datasets the site
    currently reports, not the whole list.
    """
    manifests = files("coval_bench.datasets.manifests")
    ids = sorted(
        r.name.removesuffix(".json") for r in manifests.iterdir() if r.name.endswith(".json")
    )
    return tuple(_usage_of(dataset_id) for dataset_id in ids)
