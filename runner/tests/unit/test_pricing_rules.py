# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The pricing rules: normalization, which recording prices which day, what a NewRate accepts."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from coval_bench.registries import Benchmark
from coval_bench.registries.pricing import (
    NewRate,
    PricingUnit,
    RateRecording,
    RateSpan,
    RateTimeline,
    per_1k_minutes,
    per_1m_chars,
    timelines,
)

KEY = (Benchmark.STT, "acme", "stt-1")
D = date


def _rec(
    id: int, effective_from: date, price: str | None = "0.0050", recorded_at: datetime | None = None
) -> RateRecording:
    return RateRecording(
        id=id,
        benchmark=KEY[0],
        provider=KEY[1],
        model=KEY[2],
        unit=None if price is None else PricingUnit.PER_MINUTE,
        price_usd=None if price is None else Decimal(price),
        effective_from=effective_from,
        source_url=None if price is None else "https://acme.example/pricing",
        notes=None,
        recorded_by_user_id="tests",
        recorded_by_email=None,
        recorded_at=recorded_at or datetime(2026, 9, 1, tzinfo=UTC) + timedelta(seconds=id),
    )


def _in_force(timeline: RateTimeline, day: date) -> RateSpan:
    span = timeline.in_force(day)
    assert span is not None
    return span


def test_normalization_scales_within_one_denominator_only() -> None:
    assert per_1m_chars(PricingUnit.PER_1K_CHARS, Decimal("0.030")) == Decimal("30.000")
    assert per_1m_chars(PricingUnit.PER_CHAR, Decimal("0.00003")) == Decimal("30.00000")
    assert per_1k_minutes(PricingUnit.PER_MINUTE, Decimal("0.0048")) == Decimal("4.8000")
    assert per_1k_minutes(PricingUnit.PER_HOUR, Decimal("0.45")) == Decimal("7.500000")
    assert per_1k_minutes(PricingUnit.PER_SECOND_AUDIO_IN, Decimal("0.0001")) == Decimal("6.0000")
    # Seconds of audio out and character units never convert into the other denominator.
    assert per_1m_chars(PricingUnit.PER_MINUTE, Decimal("1")) is None
    assert per_1k_minutes(PricingUnit.PER_1K_CHARS, Decimal("1")) is None
    assert per_1m_chars(PricingUnit.PER_SECOND_AUDIO_OUT, Decimal("1")) is None
    assert per_1k_minutes(PricingUnit.PER_SECOND_AUDIO_OUT, Decimal("1")) is None


def test_latest_recording_wins_spans_close_and_the_future_is_scheduled() -> None:
    rows = [
        _rec(1, D(2026, 8, 1), "0.0048"),
        _rec(2, D(2026, 9, 1), "0.0050"),
        _rec(3, D(2026, 9, 1), "0.0055"),  # a correction, recorded later
        _rec(4, D(2026, 10, 1), None),  # a delisting, still ahead
        _rec(5, D(2026, 8, 1), "1.00", recorded_at=datetime(2026, 8, 1, tzinfo=UTC)),  # earlier say
    ]
    timeline = timelines(rows)[KEY]
    assert [s.recording.id for s in timeline.spans] == [1, 3, 4]
    assert [r.id for r in timeline.superseded] == [5, 2]
    assert [s.effective_to for s in timeline.spans] == [D(2026, 9, 1), D(2026, 10, 1), None]
    assert timeline.in_force(D(2026, 7, 31)) is None
    assert _in_force(timeline, D(2026, 8, 31)).recording.id == 1
    today = _in_force(timeline, D(2026, 9, 15))
    assert today.recording.id == 3
    assert [s.recording.id for s in timeline.before(today)] == [1]
    assert [s.recording.id for s in timeline.scheduled(D(2026, 9, 15))] == [4]
    assert not _in_force(timeline, D(2026, 10, 1)).recording.priced
    assert _in_force(timeline, D(2026, 9, 15)).recording.price_per_1k_minutes == Decimal("5.5000")


def test_a_same_instant_tie_falls_to_the_later_row_and_keys_stay_apart() -> None:
    at = datetime(2026, 9, 1, tzinfo=UTC)
    other = _rec(9, D(2026, 9, 1)).model_copy(update={"model": "stt-2"})
    resolved = timelines(
        [
            _rec(2, D(2026, 9, 1), "0.2", recorded_at=at),
            _rec(1, D(2026, 9, 1), "0.1", recorded_at=at),
            other,
        ]
    )
    assert resolved[KEY].spans[0].recording.id == 2
    assert resolved[(Benchmark.STT, "acme", "stt-2")].spans[0].recording.id == 9


def _new(**overrides: Any) -> NewRate:
    fields: dict[str, Any] = {
        "benchmark": KEY[0],
        "provider": KEY[1],
        "model": KEY[2],
        "unit": "per_minute",
        "price_usd": "0.0050",
        "effective_from": D(2026, 9, 1),
        "source_url": "https://acme.example/pricing",
    }
    return NewRate.model_validate({**fields, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"price_usd": None},  # unit without a price
        {"unit": None},  # price without a unit
        {"price_usd": 0.005},  # a bare number would drop trailing zeros
        {"price_usd": "1E+2"},  # exponent form would defeat the quote-for-quote repeat check
        {"price_usd": "0"},
        {"unit": "per_1k_chars"},  # a character unit cannot bill STT
        {"source_url": None},  # a priced rate must cite its page
        {"source_url": "not a url"},
        {"provider": ""},
    ],
)
def test_new_rate_rejects_half_rates_floats_and_wrong_units(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _new(**overrides)


def test_new_rate_keeps_the_quote_and_knows_a_repeat() -> None:
    rate = _new(notes="   ")
    assert rate.notes is None and str(rate.price_usd) == "0.0050"
    assert rate.matches(_rec(1, D(2026, 9, 1), "0.0050"))
    assert not rate.matches(_rec(1, D(2026, 9, 1), "0.005"))  # same amount, different quote
    delisting = _new(unit=None, price_usd=None, source_url=None)
    assert delisting.price_usd is None and delisting.key == KEY
