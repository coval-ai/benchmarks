# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The one rule for reading the pricing log: which recording describes which day."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from coval_bench.registries import Benchmark
from coval_bench.registries.pricing.resolve import RateRecording, timelines
from coval_bench.registries.pricing.schema import PricingUnit

TODAY = date(2026, 9, 2)
KEY = (Benchmark.STT, "acme", "stt-1")


def _rec(
    id: int,
    effective_from: date,
    price: str | None = "0.006",
    *,
    unit: PricingUnit | None = PricingUnit.PER_MINUTE,
    recorded_at: datetime | None = None,
    key: tuple[Benchmark, str, str] = KEY,
) -> RateRecording:
    priced = price is not None
    return RateRecording(
        id=id,
        benchmark=key[0],
        provider=key[1],
        model=key[2],
        unit=unit if priced else None,
        price_usd=None if price is None else Decimal(price),
        effective_from=effective_from,
        source_url="https://acme.example.com/pricing" if priced else None,
        notes=None,
        recorded_by_user_id="user_test",
        recorded_by_email=None,
        recorded_at=recorded_at or datetime(2026, 9, 1, 12, 0, tzinfo=UTC) + timedelta(seconds=id),
    )


def test_no_recordings_means_no_timelines() -> None:
    assert timelines([]) == {}


def test_a_single_recording_is_an_open_span_from_its_day() -> None:
    (timeline,) = timelines([_rec(1, TODAY)]).values()
    (span,) = timeline.spans
    assert span.effective_to is None
    assert timeline.in_force(TODAY - timedelta(days=1)) is None
    assert timeline.in_force(TODAY) is span
    assert timeline.in_force(TODAY + timedelta(days=400)) is span
    assert timeline.before(span) == ()
    assert timeline.superseded == ()


def test_a_later_effective_date_closes_the_earlier_span() -> None:
    old, new = _rec(1, date(2026, 1, 1), "0.006"), _rec(2, TODAY, "0.009")
    timeline = timelines([new, old])[KEY]
    first, second = timeline.spans
    assert first.recording is old and first.effective_to == TODAY
    assert second.recording is new and second.effective_to is None
    assert timeline.in_force(date(2026, 6, 1)) is first
    assert timeline.in_force(TODAY) is second
    assert timeline.before(second) == (first,)


def test_the_latest_recording_wins_a_shared_effective_date() -> None:
    typo = _rec(1, TODAY, "0.060", recorded_at=datetime(2026, 9, 2, 9, 0, tzinfo=UTC))
    fix = _rec(2, TODAY, "0.006", recorded_at=datetime(2026, 9, 2, 9, 5, tzinfo=UTC))
    timeline = timelines([fix, typo])[KEY]
    assert [span.recording for span in timeline.spans] == [fix]
    assert timeline.superseded == (typo,)
    # The correction is not "history": the typo never described a day.
    assert timeline.before(timeline.spans[0]) == ()


def test_a_same_instant_tie_falls_to_the_later_row() -> None:
    at = datetime(2026, 9, 2, 9, 0, tzinfo=UTC)
    a, b = _rec(1, TODAY, "0.006", recorded_at=at), _rec(2, TODAY, "0.007", recorded_at=at)
    timeline = timelines([b, a])[KEY]
    assert timeline.spans[0].recording is b


def test_a_future_recording_is_scheduled_not_in_force() -> None:
    current, ahead = _rec(1, date(2026, 1, 1)), _rec(2, TODAY + timedelta(days=30), "0.009")
    timeline = timelines([current, ahead])[KEY]
    assert timeline.in_force(TODAY) is timeline.spans[0]
    assert timeline.in_force(TODAY).effective_to == ahead.effective_from  # type: ignore[union-attr]
    assert timeline.scheduled(TODAY) == (timeline.spans[1],)
    assert timeline.scheduled(TODAY + timedelta(days=30)) == ()


def test_a_delisting_is_an_unpriced_span() -> None:
    priced, gone = _rec(1, date(2026, 1, 1)), _rec(2, TODAY, None, unit=None)
    timeline = timelines([priced, gone])[KEY]
    span = timeline.in_force(TODAY)
    assert span is not None and not span.recording.priced
    assert span.recording.price_per_1k_minutes is None
    assert timeline.before(span)[0].recording.priced


def test_recordings_normalize_like_ratesheet_entries() -> None:
    hourly = _rec(1, TODAY, "0.45", unit=PricingUnit.PER_HOUR)
    assert hourly.price_per_1k_minutes == Decimal("7.5")
    assert hourly.price_per_1m_chars is None
    chars = _rec(2, TODAY, "0.030", unit=PricingUnit.PER_1K_CHARS, key=(Benchmark.TTS, "a", "m"))
    assert chars.price_per_1m_chars == Decimal("30")


def test_keys_are_kept_apart() -> None:
    other = (Benchmark.TTS, "acme", "tts-1")
    result = timelines(
        [_rec(1, TODAY), _rec(2, TODAY, "15", unit=PricingUnit.PER_1M_CHARS, key=other)]
    )
    assert set(result) == {KEY, other}
    assert result[other].spans[0].recording.price_per_1m_chars == Decimal("15")
