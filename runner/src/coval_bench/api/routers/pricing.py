# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/pricing — normalized model prices with provenance and history.

Normalization targets follow Artificial Analysis: USD per 1,000 minutes of
audio (STT, S2S) and USD per 1M characters (TTS). Token- and per-second-billed
models convert through conversion rates measured from our own runs (7-day
window); a model whose conversion isn't measurable serves its native rates
with ``normalized_usd: null`` — nothing is fabricated. History normalizes
every effective period of the append-only ``model_pricing`` table at today's
measured conversion (documented approximation on ``PriceHistoryPoint``).
Models without an effective rate are simply absent.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import psycopg.rows
import structlog
from cachetools import TTLCache
from fastapi import APIRouter, Depends, Query, Request
from psycopg_pool import AsyncConnectionPool

from coval_bench.api.cache import get_or_fill
from coval_bench.api.common import BenchmarkLiteral
from coval_bench.api.deps import get_cache, get_cache_locks, get_pool
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    ConversionOut,
    NativeRateOut,
    PriceHistoryPoint,
    PricingEntry,
    PricingResponse,
)
from coval_bench.db.models import Benchmark, PriceRow
from coval_bench.pricing.normalize import Conversion, load_conversions, normalized_price
from coval_bench.registries import MODEL_REGISTRY, ModelStatus

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["pricing"])

_UNIT_LABELS: dict[str, str] = {
    "STT": "USD per 1,000 minutes",
    "TTS": "USD per 1M characters",
    "S2S": "USD per 1,000 minutes",
}

_PRICING_SQL = """
    SELECT id, provider, model, benchmark, billing_unit, rate_usd, plan_assumption,
           effective_at, superseded_at, source_url, as_of, evidence, updated_by, created_at
    FROM benchmarks_v2.model_pricing
    WHERE benchmark = %(benchmark)s
    ORDER BY provider, model, effective_at
"""


def _history(
    benchmark: Benchmark, rows: list[PriceRow], conversion: Conversion | None
) -> list[PriceHistoryPoint]:
    """One point per effective period, normalized at today's conversion.

    Breakpoints are the distinct ``effective_at`` stamps; the rates in force at
    each breakpoint are normalized together, so token models (two rows per
    period) produce one combined point per period. By construction each period
    ends where the next begins, so ``superseded_at`` chains contiguously even
    when the units of a token pair changed at different times; only the final
    period can be open-ended.
    """
    starts = sorted({r.effective_at for r in rows})
    points: list[PriceHistoryPoint] = []
    for i, start in enumerate(starts):
        active = [
            r
            for r in rows
            if r.effective_at <= start and (r.superseded_at is None or r.superseded_at > start)
        ]
        if not active:
            continue
        if i + 1 < len(starts):
            superseded_at = starts[i + 1]
        else:
            ends = [r.superseded_at for r in active]
            superseded_at = None if any(e is None for e in ends) else max(ends)  # type: ignore[type-var, arg-type]
        normalized = normalized_price(benchmark, active, conversion)
        points.append(
            PriceHistoryPoint(
                normalized_usd=normalized.value if normalized else None,
                effective_at=start,
                superseded_at=superseded_at,
            )
        )
    return points


@router.get("/pricing", response_model=PricingResponse)
@limiter.limit("60/minute")
async def get_pricing(
    request: Request,  # required by slowapi
    benchmark: BenchmarkLiteral = Query(...),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    cache: TTLCache[Any, Any] = Depends(get_cache),
    cache_locks: defaultdict[Any, asyncio.Lock] = Depends(get_cache_locks),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> PricingResponse:
    """Normalized list prices for every active priced model of one benchmark."""

    async def fill() -> PricingResponse:
        async with pool.connection() as conn:
            conn.row_factory = psycopg.rows.dict_row
            rows = await (await conn.execute(_PRICING_SQL, {"benchmark": benchmark})).fetchall()
        conversions = await load_conversions(pool)

        by_model: dict[tuple[str, str], list[PriceRow]] = defaultdict(list)
        for r in rows:
            price = PriceRow.model_validate(dict(r))
            by_model[(price.provider, price.model)].append(price)

        bench = Benchmark(benchmark)
        visible = {
            (m.provider, m.model)
            for m in MODEL_REGISTRY
            if m.benchmark is bench
            and m.status in (ModelStatus.ACTIVE, ModelStatus.EARLY_ACCESS)
            and (m.provider, m.model) not in hidden
        }

        now = datetime.now(tz=UTC)
        entries: list[PricingEntry] = []
        for (provider, model), model_rows in sorted(by_model.items()):
            if (provider, model) not in visible:
                continue
            effective = [
                r
                for r in model_rows
                if r.effective_at <= now and (r.superseded_at is None or r.superseded_at > now)
            ]
            if not effective:
                continue
            conversion = conversions.get((provider, model, bench))
            normalized = normalized_price(bench, effective, conversion)
            entries.append(
                PricingEntry(
                    provider=provider,
                    model=model,
                    normalized_usd=normalized.value if normalized else None,
                    basis=normalized.basis if normalized else None,
                    native_rates=[
                        NativeRateOut(
                            billing_unit=str(r.billing_unit),
                            rate_usd=float(r.rate_usd),
                            plan_assumption=r.plan_assumption,
                        )
                        for r in effective
                    ],
                    conversion=(
                        ConversionOut(
                            in_tokens_per_min=conversion.in_tokens_per_min,
                            out_tokens_per_min=conversion.out_tokens_per_min,
                            chars_per_sec=conversion.chars_per_sec,
                            sample_count=conversion.sample_count,
                            window=conversion.window,
                        )
                        if conversion
                        else None
                    ),
                    as_of=max(r.as_of for r in effective),
                    source_url=effective[0].source_url,
                    history=_history(bench, model_rows, conversion),
                )
            )
        return PricingResponse(
            benchmark=benchmark, unit_label=_UNIT_LABELS[benchmark], entries=entries
        )

    cache_key = ("pricing", benchmark, tuple(sorted(hidden)))
    response, _ = await get_or_fill(cache, cache_locks, cache_key, fill)
    return response
