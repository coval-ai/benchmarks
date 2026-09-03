# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Which rate was in force when: the one place the pricing log is interpreted.

``benchmarks_v2.pricing_rates`` is append-only. Every row is a *recording*: on
``recorded_at`` someone (an admin, a migration, the ratesheet sync) said that
from ``effective_from`` a model costs ``price_usd`` per ``unit`` — or, with both
null, that no public rate is known from that day. Rows are never updated or
deleted, so the table is the full history and this module is the rule that
reads it. Two facts fall out of the rule, and everything that serves a price
must go through here so they hold everywhere:

* **For one effective date, the latest recording wins.** Recording a rate again
  for a date that already has one is how a mistake is corrected — the earlier
  recording stays in the log but stops describing that date.
* **The rate in force on a day is the winning recording with the greatest
  effective date at or before it.** A recording dated in the future is
  scheduled: it prices nothing until its day arrives, and a public read never
  sees it.

Per key the winning recordings form a *timeline* of spans, each closed by the
next span's effective date. A span whose recording is unpriced is a delisting:
the model is known to have no public rate over that span.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal

from pydantic import BaseModel, field_validator

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.pricing.schema import PricingUnit, per_1k_minutes, per_1m_chars

RateKey = tuple[Benchmark, str, str]


class RateRecording(BaseModel, frozen=True):
    """One row of ``benchmarks_v2.pricing_rates``."""

    id: int
    benchmark: Benchmark
    provider: str
    model: str
    # Both set (a published rate) or both None (no public rate from this date).
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
        # The driver hands back timestamptz in the session's zone; serve one
        # zone everywhere so two API instances never disagree on the stamp.
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

    def covers(self, day: date) -> bool:
        return self.effective_from <= day and (self.effective_to is None or day < self.effective_to)


class RateTimeline(BaseModel, frozen=True):
    """One model's spans, ascending by effective date, gapless from the first."""

    key: RateKey
    spans: tuple[RateSpan, ...]
    # Recordings that lost to a later one for the same date: kept for the
    # admin's audit view, invisible to the public reads.
    superseded: tuple[RateRecording, ...]

    def in_force(self, day: date) -> RateSpan | None:
        """The span covering *day*, or None before the first recording."""
        starts = [span.effective_from for span in self.spans]
        index = bisect_right(starts, day) - 1
        return self.spans[index] if index >= 0 else None

    def before(self, span: RateSpan) -> tuple[RateSpan, ...]:
        """Every span that ended before *span* began — the public history."""
        return tuple(s for s in self.spans if s.effective_from < span.effective_from)

    def scheduled(self, day: date) -> tuple[RateSpan, ...]:
        """Spans whose effective date is still ahead of *day*."""
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
    # Later recordings win; the id breaks a same-instant tie the way the
    # identity column already ordered them.
    for row in sorted(rows, key=lambda r: (r.recorded_at, r.id)):
        loser = winners.get(row.effective_from)
        if loser is not None:
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
