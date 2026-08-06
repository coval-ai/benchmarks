# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Report ACTIVE / EARLY_ACCESS models missing an effective ``model_pricing`` rate.

Mirrors ``check_arena_keys.py``: exits non-zero with the offending list so it
can gate CI or be run manually after registry changes. Known-unpriced models
(no published rate anywhere — the seed script's gap list) still fail here:
the failure IS the staleness signal to re-research them.

Usage: DATABASE_URL=... uv run python scripts/check_pricing_coverage.py
"""

from __future__ import annotations

import asyncio

from coval_bench.config import get_settings
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.pricing import PricingStore
from coval_bench.registries.models import MODEL_REGISTRY, ModelStatus


async def _unpriced() -> list[str]:
    async with lifespan_pool(get_settings()) as pool:
        store = PricingStore(pool)
        await store.load_cache()
        return [
            f"{m.benchmark}:{m.provider}/{m.model}"
            for m in MODEL_REGISTRY
            if m.status in (ModelStatus.ACTIVE, ModelStatus.EARLY_ACCESS)
            and not any(
                r.benchmark is m.benchmark
                for r in store.effective_rates_cached(m.provider, m.model)
            )
        ]


def main() -> int:
    missing = asyncio.run(_unpriced())
    if missing:
        print(
            f"ERROR: {len(missing)} active models have no effective pricing row:\n  "
            + "\n  ".join(missing)
            + "\nAdd rates via scripts/seed_pricing.py (with source_url + as_of), "
            "or pause the model."
        )
        return 1
    print("OK: every active model has an effective pricing row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
