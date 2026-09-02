# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/pricing — the rates in force on a day, with their history.

The response carries one entry per listed model that has ever had a rate
recorded: the rate in force on ``as_of`` (today by default), the earlier rates
it succeeded, and the packaged datasets' usage (minutes of audio in,
characters spoken), so a client can turn a rate into a benchmark-pass cost by
pure arithmetic — against the datasets the site currently reports, since the
usage list also keeps retired datasets whose artifacts still reference them.

Rates live in ``benchmarks_v2.pricing_rates`` and change through the admin
API (see ``admin_pricing.py``), never by deploy; ``registries.pricing.resolve``
is the one rule for which recording describes which day. A rate scheduled for
a future day prices nothing until then, and ``as_of`` may not be in the future
either — the site never shows a price that is not yet in force.

Visibility follows the model roster: a rate serves only while the roster
lists its model — uncollected models included, since the site shows them
greyed out — and unpublished models are hidden unless the caller's bearer
token clears them, the same embargo every other data endpoint applies.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import capture_api_event, get_models, get_pool, get_posthog
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    DatasetUsageOut,
    PricingRateOut,
    PricingRateSpanOut,
    PricingRegistryResponse,
)
from coval_bench.datasets.usage import dataset_usage
from coval_bench.db.pricing_store import PricingStore
from coval_bench.registries import RegisteredModel
from coval_bench.registries.pricing.resolve import RateSpan, RateTimeline, timelines

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["pricing"])


def _span_out(span: RateSpan, today: date) -> PricingRateSpanOut:
    r = span.recording
    per_chars = r.price_per_1m_chars
    per_minutes = r.price_per_1k_minutes
    # A span closed by a *scheduled* recording is served open: the public read
    # never shows a rate before its day, so it must not name that day either.
    effective_to = span.effective_to
    if effective_to is not None and effective_to > today:
        effective_to = None
    return PricingRateSpanOut(
        unit=None if r.unit is None else str(r.unit),
        price_usd=r.price_usd,
        price_per_1m_chars=float(per_chars) if per_chars is not None else None,
        price_per_1k_minutes=float(per_minutes) if per_minutes is not None else None,
        effective_from=span.effective_from,
        effective_to=effective_to,
        source_url=r.source_url,
    )


def _rate_out(timeline: RateTimeline, span: RateSpan, today: date) -> PricingRateOut:
    r = span.recording
    return PricingRateOut(
        **_span_out(span, today).model_dump(),
        benchmark=r.benchmark,
        provider=r.provider,
        model=r.model,
        notes=r.notes,
        recorded_at=r.recorded_at,
        history=[_span_out(s, today) for s in timeline.before(span)],
    )


# The manifests are immutable for the life of the process, so their usage is
# shaped once at import — a startup cost instead of the first request's turn.
_USAGE: tuple[DatasetUsageOut, ...] = tuple(
    DatasetUsageOut.model_validate(u, from_attributes=True) for u in dataset_usage()
)


@router.get("/pricing", response_model=PricingRegistryResponse)
@limiter.limit("60/minute")
async def get_pricing_registry(
    request: Request,
    as_of: date | None = Query(
        default=None,
        description="The day to price as of (UTC); defaults to today and may not be later.",
    ),
    posthog_client: Posthog | None = Depends(get_posthog),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
    models: Sequence[RegisteredModel] = Depends(get_models),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> PricingRegistryResponse:
    """Return every rate in force on ``as_of`` plus per-dataset usage figures."""
    today = datetime.now(UTC).date()
    day = today if as_of is None else as_of
    if day > today:
        raise HTTPException(422, f"as_of may not be later than today ({today.isoformat()} UTC)")
    try:
        recordings = await PricingStore(pool).recordings()
    except Exception as exc:
        logger.error("pricing_log_unavailable", exc_info=True)
        raise HTTPException(503, "the pricing log is unavailable") from exc

    roster = {(m.benchmark, m.provider, m.model) for m in models}
    rates: list[PricingRateOut] = []
    for key, timeline in sorted(timelines(recordings).items()):
        if key not in roster or (key[1], key[2]) in hidden:
            continue
        span = timeline.in_force(day)
        if span is None:
            continue
        rates.append(_rate_out(timeline, span, today))

    capture_api_event(
        posthog_client,
        "pricing_listed",
        {
            "rate_count": len(rates),
            "as_of_requested": as_of is not None,
            "$process_person_profile": False,
        },
    )
    return PricingRegistryResponse(as_of=day, rates=rates, usage=list(_USAGE))
