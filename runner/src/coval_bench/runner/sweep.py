# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality sweep: every item of every dataset through a model, once per model version.

This opens the sweeps: one for each model version and dataset version that has
none. Nothing is sent to a provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coval_bench.datasets.loader import dataset_manifest
from coval_bench.datasets.suite import SWEEP_STT_DATASETS
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.registry_store import RegistryStore, _registered
from coval_bench.db.sweep_store import ensure_sweeps
from coval_bench.registries import Benchmark
from coval_bench.runner.orchestrator import _select_models

if TYPE_CHECKING:
    from coval_bench.config import Settings


async def open_sweeps(*, settings: Settings, only: frozenset[str] | None = None) -> tuple[int, int]:
    """Open any missing sweeps for the collected shared STT models.

    Returns (models looked at, sweeps opened).
    """
    async with lifespan_pool(settings) as pool:
        records = [
            r for r in await RegistryStore(pool).list_models() if r.modality is Benchmark.STT
        ]
        ids = {(r.provider, r.model): r.id for r in records}
        models, _ = _select_models(
            [_registered(r) for r in records],
            [],
            source="shared",
            benchmark_kind="stt",
            only=only,
        )
        versions = {
            dataset_id: dataset_manifest(dataset_id).version for dataset_id in SWEEP_STT_DATASETS
        }
        opened = await ensure_sweeps(
            pool,
            [
                (ids[(m.provider, m.model)], dataset_id, version)
                for m in models
                for dataset_id, version in versions.items()
            ],
        )
    return len(models), opened
