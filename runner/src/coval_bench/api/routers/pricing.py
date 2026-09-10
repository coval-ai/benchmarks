# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/pricing — each listed model's rate in force on a day, with its history.

Rates live in ``benchmarks_v2.pricing_rates`` and change through the admin API
(``admin_pricing.py``), never by deploy; ``registries.pricing`` is the one rule
for which recording describes which day. ``as_of`` defaults to today (UTC) and
may not be later: a change scheduled for a future day is not public until then,
its date included. Visibility follows the model roster and the same early-access
embargo every other data endpoint applies.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

import psycopg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import capture_api_event, get_models, get_pool, get_posthog
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import PricingRateOut, PricingRateSpanOut, PricingRegistryResponse
from coval_bench.db.pricing_store import PricingStore
from coval_bench.registries import RegisteredModel
from coval_bench.registries.pricing import RateRecording, RateSpan, timelines

logger = structlog.get_logger("coval_bench.api")
router = APIRouter(tags=["pricing"])

PRICING_LOG_UNAVAILABLE = HTTPException(503, "the pricing log is unavailable")


def span_fields(recording: RateRecording, effective_to: date | None = None) -> dict[str, Any]:
    """The ``PricingRateSpanOut`` fields of a recording, shared with the admin router."""
    chars, minutes = recording.price_per_1m_chars, recording.price_per_1k_minutes
    return {
        "unit": None if recording.unit is None else str(recording.unit),
        "price_usd": recording.price_usd,
        "price_per_1m_chars": None if chars is None else float(chars),
        "price_per_1k_minutes": None if minutes is None else float(minutes),
        "effective_from": recording.effective_from,
        "effective_to": effective_to,
        "source_url": recording.source_url,
    }


def _public_span(span: RateSpan, today: date) -> dict[str, Any]:
    # A span closed by a scheduled change is still open as far as the public knows.
    to = span.effective_to
    return span_fields(span.recording, None if to is not None and to > today else to)


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
    """Return every rate in force on ``as_of``, with the earlier spans as history."""
    today = datetime.now(UTC).date()
    day = as_of or today
    if day > today:
        raise HTTPException(422, f"as_of may not be later than today ({today.isoformat()} UTC)")
    try:
        recordings = await PricingStore(pool).recordings()
    except psycopg.Error as exc:
        logger.warning("pricing_log_unavailable", error=str(exc))
        raise PRICING_LOG_UNAVAILABLE from exc
    roster = {(m.benchmark, m.provider, m.model) for m in models}
    rates: list[PricingRateOut] = []
    for key, timeline in sorted(timelines(recordings).items()):
        span = timeline.in_force(day)
        if span is None or key not in roster or (key[1], key[2]) in hidden:
            continue
        rates.append(
            PricingRateOut(
                **_public_span(span, today),
                benchmark=key[0],
                provider=key[1],
                model=key[2],
                notes=span.recording.notes,
                recorded_at=span.recording.recorded_at,
                history=[
                    PricingRateSpanOut(**_public_span(s, today)) for s in timeline.before(span)
                ],
            )
        )
    capture_api_event(
        posthog_client,
        "pricing_listed",
        {
            "rate_count": len(rates),
            "as_of_requested": as_of is not None,
            "$process_person_profile": False,
        },
    )
    return PricingRegistryResponse(as_of=day, rates=rates)
