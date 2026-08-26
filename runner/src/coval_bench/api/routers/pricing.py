# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/pricing — the packaged pricing registry, served for the public page.

The response carries every rate in force today plus the packaged datasets'
usage (minutes of audio in, characters spoken), so a client can turn a rate
into a benchmark-pass cost by pure arithmetic. Rates change by pull request
against the JSON ratesheets (see CONTRIBUTING.md), never at runtime, and a
rate whose ``effective_from`` is still ahead prices nothing until that day.

Visibility follows the model roster: a rate serves only while the roster
lists its model — uncollected models included, since the site shows them
greyed out — and unpublished models are hidden unless the caller's bearer
token clears them, the same embargo every other data endpoint applies. The
roster read is this endpoint's only database hit; the rates ship in the
package.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from fastapi import APIRouter, Depends
from starlette.requests import Request

from coval_bench.api.deps import get_models
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import DatasetUsageOut, PricingRateOut, PricingRegistryResponse
from coval_bench.datasets.usage import dataset_usage
from coval_bench.registries import RegisteredModel
from coval_bench.registries.pricing import PRICING, PricingEntry

router = APIRouter(tags=["pricing"])


def _rate_out(entry: PricingEntry) -> PricingRateOut:
    per_chars = entry.price_per_1m_chars
    per_minutes = entry.price_per_1k_minutes
    return PricingRateOut(
        benchmark=entry.benchmark,
        provider=entry.provider,
        model=entry.model,
        unit=entry.unit,
        price_usd=entry.price_usd,
        price_per_1m_chars=float(per_chars) if per_chars is not None else None,
        price_per_1k_minutes=float(per_minutes) if per_minutes is not None else None,
        effective_from=entry.effective_from,
        source_url=str(entry.source_url),
        notes=entry.notes,
    )


@router.get("/pricing", response_model=PricingRegistryResponse)
@limiter.limit("60/minute")
async def get_pricing_registry(
    request: Request,
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
    models: Sequence[RegisteredModel] = Depends(get_models),
) -> PricingRegistryResponse:
    """Return every rate in force today plus per-dataset usage figures."""
    today = date.today()
    roster = {(m.benchmark, m.provider, m.model) for m in models}
    return PricingRegistryResponse(
        as_of=today,
        rates=[
            _rate_out(entry)
            for key, entry in sorted(PRICING.items())
            if entry.effective_from <= today and key in roster and (key[1], key[2]) not in hidden
        ],
        usage=[DatasetUsageOut.model_validate(u, from_attributes=True) for u in dataset_usage()],
    )
