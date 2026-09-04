# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Admin pricing endpoints: read the whole pricing log, append to it.

Coval-org only (``require_coval_admin``) and every response is private: the log
names staff and may price models still under embargo. There is no PATCH and no
DELETE — a mistake is corrected by recording the right value for the same
effective date, which the log keeps beside the wrong one and marks superseded.
"""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Response
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from coval_bench.api.clerk import CovalAdmin
from coval_bench.api.deps import get_pool, require_coval_admin
from coval_bench.api.internal import never_shared
from coval_bench.api.routers.pricing import PRICING_LOG_UNAVAILABLE, span_fields
from coval_bench.api.schemas import (
    AdminModelPricingOut,
    AdminPricingResponse,
    AdminRateCreate,
    AdminRateRecordingOut,
)
from coval_bench.db.pricing_store import PricingStore
from coval_bench.registries.pricing import (
    NewRate,
    RateKey,
    RateRecording,
    RateTimeline,
    timelines,
)

router = APIRouter(tags=["admin"], dependencies=[Depends(require_coval_admin)])

MAX_SCHEDULE_AHEAD = timedelta(days=366)
EARLIEST_EFFECTIVE = date(2020, 1, 1)


async def _guarded[T](call: Awaitable[T]) -> T:
    """Run a store call; a missing table or grant is a 503, not a traceback."""
    try:
        return await call
    except psycopg.Error as exc:
        raise PRICING_LOG_UNAVAILABLE from exc


def _recording_out(recording: RateRecording, superseded: bool) -> AdminRateRecordingOut:
    return AdminRateRecordingOut(
        **span_fields(recording),
        id=recording.id,
        notes=recording.notes,
        recorded_by_user_id=recording.recorded_by_user_id,
        recorded_by_email=recording.recorded_by_email,
        recorded_at=recording.recorded_at,
        superseded=superseded,
    )


def _model_out(
    timeline: RateTimeline, recordings: list[RateRecording], today: date
) -> AdminModelPricingOut:
    superseded = {r.id for r in timeline.superseded}
    current = timeline.in_force(today)
    return AdminModelPricingOut(
        benchmark=timeline.key[0],
        provider=timeline.key[1],
        model=timeline.key[2],
        current=None if current is None else _recording_out(current.recording, False),
        scheduled=[_recording_out(s.recording, False) for s in timeline.scheduled(today)],
        recordings=[
            _recording_out(r, r.id in superseded)
            for r in sorted(recordings, key=lambda r: (r.recorded_at, r.id), reverse=True)
        ],
    )


@router.get("/admin/pricing", response_model=AdminPricingResponse)
async def list_admin_pricing(
    response: Response, pool: AsyncConnectionPool[Any] = Depends(get_pool)
) -> AdminPricingResponse:
    """Every model with a pricing log entry: current, scheduled, and the full audit trail."""
    never_shared(response)
    today = datetime.now(UTC).date()
    recordings = await _guarded(PricingStore(pool).recordings())
    by_key: dict[RateKey, list[RateRecording]] = {}
    for recording in recordings:
        by_key.setdefault(recording.key, []).append(recording)
    return AdminPricingResponse(
        as_of=today,
        models=[
            _model_out(timeline, by_key[key], today)
            for key, timeline in sorted(timelines(recordings).items())
        ],
    )


@router.post("/admin/pricing", response_model=AdminModelPricingOut, status_code=201)
async def record_admin_rate(
    body: AdminRateCreate,
    response: Response,
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> AdminModelPricingOut:
    """Append a recording; 201 when it wrote, 200 when the date already said exactly this."""
    never_shared(response)
    today = datetime.now(UTC).date()
    try:
        new = NewRate.model_validate(
            {**body.model_dump(), "effective_from": body.effective_from or today}
        )
    except ValidationError as exc:
        messages = (str(e["msg"]).removeprefix("Value error, ") for e in exc.errors())
        raise HTTPException(422, "; ".join(messages)) from exc
    if new.effective_from > today + MAX_SCHEDULE_AHEAD:
        raise HTTPException(422, "effective_from may be at most a year ahead")
    if new.effective_from < EARLIEST_EFFECTIVE:
        raise HTTPException(422, f"effective_from may not precede {EARLIEST_EFFECTIVE.isoformat()}")
    store = PricingStore(pool)
    if not await _guarded(store.model_exists(new.key)):
        raise HTTPException(
            422,
            f"no registered model {new.benchmark}/{new.provider}/{new.model}; "
            "add it to the registry before pricing it",
        )
    _, inserted = await _guarded(store.record(new, user_id=admin.user_id, email=admin.email))
    if not inserted:
        response.status_code = 200
    recordings = await _guarded(store.recordings_for(new.key))
    return _model_out(timelines(recordings)[new.key], recordings, today)
