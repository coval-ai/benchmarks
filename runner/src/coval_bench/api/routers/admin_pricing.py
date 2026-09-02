# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Admin pricing endpoints: read the whole pricing log, append to it.

Every route requires a coval-org Clerk session (``require_coval_admin``) and
every response is marked private: the log names staff and may price models
that are still embargoed. There is no PATCH and no DELETE — a price is changed
by recording a new one, and a mistake is corrected by recording the right
value for the same effective date, which the log keeps beside the wrong one.

Writes accept exactly what a ratesheet entry may say (``NewRate`` runs the
entry through ``PricingEntry``), for a model the registry lists in any state —
pricing a hidden model ahead of its launch is expected — dated no more than a
year ahead, so a slipped keystroke cannot schedule a rate for 2062.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from coval_bench.api.clerk import CovalAdmin
from coval_bench.api.deps import get_pool, require_coval_admin
from coval_bench.api.internal import never_shared
from coval_bench.api.schemas import (
    AdminModelPricingOut,
    AdminPricingResponse,
    AdminRateCreate,
    AdminRateRecordingOut,
)
from coval_bench.db.pricing_store import NewRate, PricingStore
from coval_bench.registries.pricing.resolve import (
    RateKey,
    RateRecording,
    RateTimeline,
    timelines,
)

router = APIRouter(tags=["admin"], dependencies=[Depends(require_coval_admin)])

# How far ahead a rate may be scheduled. Providers announce changes weeks out,
# not years; anything further is far more likely a typo than a plan.
MAX_SCHEDULE_AHEAD = timedelta(days=366)
# And how far back one may be dated: the oldest model we benchmark postdates this.
EARLIEST_EFFECTIVE = date(2020, 1, 1)


def _recording_out(recording: RateRecording, superseded: bool) -> AdminRateRecordingOut:
    per_chars = recording.price_per_1m_chars
    per_minutes = recording.price_per_1k_minutes
    return AdminRateRecordingOut(
        id=recording.id,
        unit=None if recording.unit is None else str(recording.unit),
        price_usd=recording.price_usd,
        price_per_1m_chars=float(per_chars) if per_chars is not None else None,
        price_per_1k_minutes=float(per_minutes) if per_minutes is not None else None,
        effective_from=recording.effective_from,
        source_url=recording.source_url,
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


def _today() -> date:
    return datetime.now(UTC).date()


@router.get("/admin/pricing", response_model=AdminPricingResponse)
async def list_admin_pricing(
    response: Response,
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> AdminPricingResponse:
    """Every model with a pricing log entry: current, scheduled, and the full audit trail."""
    never_shared(response)
    today = _today()
    recordings = await PricingStore(pool).recordings()
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


def _messages(exc: ValidationError) -> str:
    return "; ".join(str(error["msg"]).removeprefix("Value error, ") for error in exc.errors())


@router.post("/admin/pricing", response_model=AdminModelPricingOut, status_code=201)
async def record_admin_rate(
    body: AdminRateCreate,
    response: Response,
    admin: CovalAdmin = Depends(require_coval_admin),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> AdminModelPricingOut:
    """Append a recording; 201 when it wrote, 200 when the date already said exactly this."""
    never_shared(response)
    today = _today()
    try:
        new = NewRate.model_validate(
            {**body.model_dump(), "effective_from": body.effective_from or today}
        )
    except ValidationError as exc:
        raise HTTPException(422, _messages(exc)) from exc
    if new.effective_from > today + MAX_SCHEDULE_AHEAD:
        raise HTTPException(422, "effective_from may be at most a year ahead")
    if new.effective_from < EARLIEST_EFFECTIVE:
        raise HTTPException(422, f"effective_from may not precede {EARLIEST_EFFECTIVE.isoformat()}")

    store = PricingStore(pool)
    if not await store.model_exists(new.key):
        raise HTTPException(
            422,
            f"no registered model {new.benchmark}/{new.provider}/{new.model}; "
            "add it to the registry before pricing it",
        )
    _, inserted = await store.record(new, user_id=admin.user_id, email=admin.email)
    if not inserted:
        response.status_code = 200
    recordings = await store.recordings_for(new.key)
    return _model_out(timelines(recordings)[new.key], recordings, today)
