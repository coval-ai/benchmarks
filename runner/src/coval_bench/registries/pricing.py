# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The pricing log's rules: units, normalization, and which recording prices which day.

``benchmarks_v2.pricing_rates`` is append-only. A row is a *recording*: on
``recorded_at`` someone said that from ``effective_from`` a model costs
``price_usd`` per ``unit`` — or, with both null, that no public rate is known
from that day. For one effective date the latest recording wins (that is how a
mistake is corrected); the rate in force on a day is the winning recording with
the greatest effective date at or before it, so a future-dated recording is
scheduled and prices nothing until its day arrives.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from coval_bench.registries.benchmarks import Benchmark

RateKey = tuple[Benchmark, str, str]


class PricingUnit(StrEnum):
    """The native unit a provider publishes in.

    Character units bill TTS input, duration units bill STT input. Each normalizes
    within its own denominator; nothing converts across the two, which would take
    an assumed speaking rate.
    """

    PER_1M_CHARS = "per_1m_chars"
    PER_1K_CHARS = "per_1k_chars"
    PER_CHAR = "per_char"
    PER_SECOND_AUDIO_OUT = "per_second_audio_out"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_SECOND_AUDIO_IN = "per_second_audio_in"


UNITS_BY_BENCHMARK: dict[Benchmark, frozenset[PricingUnit]] = {
    Benchmark.TTS: frozenset(
        {
            PricingUnit.PER_1M_CHARS,
            PricingUnit.PER_1K_CHARS,
            PricingUnit.PER_CHAR,
            PricingUnit.PER_SECOND_AUDIO_OUT,
        }
    ),
    Benchmark.STT: frozenset(
        {PricingUnit.PER_MINUTE, PricingUnit.PER_HOUR, PricingUnit.PER_SECOND_AUDIO_IN}
    ),
}

# Exact multipliers to the site's two figures. per_hour divides, so it rounds
# half-up to a millionth; per_second_audio_out has no equivalent in either.
_TO_1M_CHARS = {
    PricingUnit.PER_1M_CHARS: 1,
    PricingUnit.PER_1K_CHARS: 1000,
    PricingUnit.PER_CHAR: 1_000_000,
}
_TO_1K_MINUTES = {PricingUnit.PER_MINUTE: 1000, PricingUnit.PER_SECOND_AUDIO_IN: 60_000}


def per_1m_chars(unit: PricingUnit, price_usd: Decimal) -> Decimal | None:
    """USD per 1M characters, or None when the unit is not character-billed."""
    factor = _TO_1M_CHARS.get(unit)
    return None if factor is None else price_usd * factor


def per_1k_minutes(unit: PricingUnit, price_usd: Decimal) -> Decimal | None:
    """USD per 1,000 minutes of input audio, or None when the unit does not bill audio in."""
    if unit is PricingUnit.PER_HOUR:
        return (price_usd * 1000 / 60).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    factor = _TO_1K_MINUTES.get(unit)
    return None if factor is None else price_usd * factor


class RateRecording(BaseModel, frozen=True):
    """One row of ``benchmarks_v2.pricing_rates``."""

    id: int
    benchmark: Benchmark
    provider: str
    model: str
    unit: PricingUnit | None
    price_usd: Decimal | None
    effective_from: date
    source_url: str | None
    notes: str | None
    recorded_by_user_id: str
    recorded_by_email: str | None
    recorded_at: datetime

    @field_validator("recorded_at")
    @classmethod
    def _in_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)

    @property
    def key(self) -> RateKey:
        return (self.benchmark, self.provider, self.model)

    @property
    def priced(self) -> bool:
        return self.price_usd is not None

    @property
    def price_per_1m_chars(self) -> Decimal | None:
        if self.unit is None or self.price_usd is None:
            return None
        return per_1m_chars(self.unit, self.price_usd)

    @property
    def price_per_1k_minutes(self) -> Decimal | None:
        if self.unit is None or self.price_usd is None:
            return None
        return per_1k_minutes(self.unit, self.price_usd)


class RateSpan(BaseModel, frozen=True):
    """A winning recording and the day the next one takes over (exclusive)."""

    recording: RateRecording
    effective_to: date | None

    @property
    def effective_from(self) -> date:
        return self.recording.effective_from


class RateTimeline(BaseModel, frozen=True):
    """One model's spans, ascending by effective date, plus the recordings they replaced."""

    key: RateKey
    spans: tuple[RateSpan, ...]
    superseded: tuple[RateRecording, ...]

    def in_force(self, day: date) -> RateSpan | None:
        """The span covering *day*, or None before the first recording."""
        index = bisect_right([span.effective_from for span in self.spans], day) - 1
        return self.spans[index] if index >= 0 else None

    def before(self, span: RateSpan) -> tuple[RateSpan, ...]:
        """Every span that ended before *span* began: the public history."""
        return tuple(s for s in self.spans if s.effective_from < span.effective_from)

    def scheduled(self, day: date) -> tuple[RateSpan, ...]:
        return tuple(s for s in self.spans if s.effective_from > day)


def timelines(recordings: Iterable[RateRecording]) -> dict[RateKey, RateTimeline]:
    """Resolve every recording into per-model timelines."""
    by_key: dict[RateKey, list[RateRecording]] = defaultdict(list)
    for recording in recordings:
        by_key[recording.key].append(recording)
    return {key: _timeline(key, rows) for key, rows in by_key.items()}


def _timeline(key: RateKey, rows: list[RateRecording]) -> RateTimeline:
    winners: dict[date, RateRecording] = {}
    superseded: list[RateRecording] = []
    for row in sorted(rows, key=lambda r: (r.recorded_at, r.id)):
        if (loser := winners.get(row.effective_from)) is not None:
            superseded.append(loser)
        winners[row.effective_from] = row
    ordered = [winners[day] for day in sorted(winners)]
    spans = tuple(
        RateSpan(
            recording=recording,
            effective_to=ordered[i + 1].effective_from if i + 1 < len(ordered) else None,
        )
        for i, recording in enumerate(ordered)
    )
    return RateTimeline(key=key, spans=spans, superseded=tuple(superseded))


class NewRate(BaseModel, frozen=True, extra="forbid"):
    """A rate as an admin states it, before the log assigns identity and time.

    ``unit`` and ``price_usd`` come together (a published rate, which must cite the
    page that prints it) or not at all (no known public rate from ``effective_from``).
    The price is a decimal string so the quoted figure survives exactly.
    """

    benchmark: Benchmark
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    unit: PricingUnit | None = None
    price_usd: Annotated[Decimal, Field(gt=0)] | None = None
    effective_from: date
    source_url: HttpUrl | None = None
    notes: str | None = None

    @field_validator("price_usd", mode="before")
    @classmethod
    def _price_written_as_string(cls, value: object) -> object:
        if isinstance(value, float | int) or (isinstance(value, str) and "e" in value.lower()):
            raise ValueError('write price_usd as a plain decimal string, e.g. "0.20"')
        return value

    @field_validator("notes", mode="before")
    @classmethod
    def _blank_notes_are_none(cls, value: object) -> object:
        return (value.strip() or None) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _priced_or_delisted(self) -> NewRate:
        if (self.unit is None) != (self.price_usd is None):
            raise ValueError(
                "a rate needs both a unit and a price; "
                "a delisting (no known public rate) has neither"
            )
        if self.unit is not None:
            if self.unit not in UNITS_BY_BENCHMARK.get(self.benchmark, frozenset()):
                raise ValueError(f"{self.unit} does not bill {self.benchmark} input")
            if self.source_url is None:
                raise ValueError("a priced rate must cite the public page that prints the figure")
        return self

    @property
    def key(self) -> RateKey:
        return (self.benchmark, self.provider, self.model)

    def matches(self, recording: RateRecording) -> bool:
        """Whether *recording* already states this rate for this date, quote for quote."""
        return (
            self.key == recording.key
            and self.effective_from == recording.effective_from
            and self.unit == recording.unit
            and str(self.price_usd) == str(recording.price_usd)
            and (self.source_url and str(self.source_url) or None) == recording.source_url
            and self.notes == recording.notes
        )
