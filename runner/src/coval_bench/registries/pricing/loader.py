# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Loader for the pricing registry: per-provider JSON, validated on import.

Files ship inside the wheel under ``pricing/data/tts/<provider>.json`` and are
loaded via ``importlib.resources``, mirroring the dataset manifests. A file
may only price its own provider's models, every entry must match a
``MODEL_REGISTRY`` row, and duplicate ``(benchmark, provider, model)`` keys
raise — all at import time, so a bad PR can never reach the API. Missing
pricing for a model is fine: it renders as absent, never $0.
"""

from __future__ import annotations

from collections import Counter
from importlib.resources import files

from pydantic import TypeAdapter

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import MODEL_REGISTRY
from coval_bench.registries.pricing.schema import PricingEntry

_ENTRIES = TypeAdapter(list[PricingEntry])


def _load_entries() -> list[PricingEntry]:
    data_dir = files("coval_bench.registries.pricing") / "data" / "tts"
    entries: list[PricingEntry] = []
    for resource in sorted(data_dir.iterdir(), key=lambda r: r.name):
        if not resource.name.endswith(".json"):
            continue
        provider = resource.name.removesuffix(".json")
        loaded = _ENTRIES.validate_json(resource.read_bytes())
        strays = sorted({e.provider for e in loaded} - {provider})
        if strays:
            raise RuntimeError(
                f"pricing/data/tts/{resource.name} prices other providers: {', '.join(strays)}"
            )
        entries.extend(loaded)
    return entries


def index_pricing(
    entries: list[PricingEntry],
) -> dict[tuple[Benchmark, str, str], PricingEntry]:
    """Index entries by registry key, rejecting duplicates and unknown models."""
    key_counts = Counter((e.benchmark, e.provider, e.model) for e in entries)
    dupes = sorted(f"{b}:{p}/{m}" for (b, p, m), n in key_counts.items() if n > 1)
    if dupes:
        raise RuntimeError(f"pricing registry contains duplicate entries: {', '.join(dupes)}")
    registered = {(m.benchmark, m.provider, m.model) for m in MODEL_REGISTRY}
    unknown = sorted(
        f"{e.benchmark}:{e.provider}/{e.model}"
        for e in entries
        if (e.benchmark, e.provider, e.model) not in registered
    )
    if unknown:
        raise RuntimeError(
            f"pricing registry entries match no MODEL_REGISTRY row: {', '.join(unknown)}"
        )
    return {(e.benchmark, e.provider, e.model): e for e in entries}


PRICING: dict[tuple[Benchmark, str, str], PricingEntry] = index_pricing(_load_entries())
