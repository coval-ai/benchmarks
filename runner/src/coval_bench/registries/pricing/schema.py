# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Schema for the provider-editable pricing registry.

Entries live in per-provider JSON files under ``pricing/data/<benchmark>/``
and carry the provider's *native* published unit; normalization to the two
figures the API serves ($ per 1M characters for TTS, $ per 1,000 minutes for
STT) happens here, never in the files.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from coval_bench.registries.benchmarks import Benchmark

# Scale for the one normalization that divides ($/hour → $/1,000 minutes, which
# is × 1000 ÷ 60 and rarely terminates). Rounding it deliberately beats letting
# the ambient Decimal context decide where the tail falls: a millionth of a
# dollar per 1,000 minutes is finer than any published rate's own precision, and
# the served figure is then identical everywhere. Every other unit scales by a
# power of ten and stays exact.
_DIVIDED_SCALE = Decimal("0.000001")


class PricingUnit(StrEnum):
    """The native unit a provider publishes its rate in.

    Character units bill TTS input, duration units bill STT input. Each
    normalizes within its own denominator; nothing converts across the two,
    which would take an assumed speaking rate.
    """

    PER_1M_CHARS = "per_1m_chars"
    PER_1K_CHARS = "per_1k_chars"
    PER_CHAR = "per_char"
    PER_SECOND_AUDIO_OUT = "per_second_audio_out"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_SECOND_AUDIO_IN = "per_second_audio_in"


# Which units bill which benchmark's input — the pairing CONTRIBUTING's table
# states, enforced here so a mismatched entry fails at import with the entry
# named, not later as a bare assert in a data test.
_UNITS_BY_BENCHMARK: dict[Benchmark, frozenset[PricingUnit]] = {
    Benchmark.TTS: frozenset(
        {
            PricingUnit.PER_1M_CHARS,
            PricingUnit.PER_1K_CHARS,
            PricingUnit.PER_CHAR,
            PricingUnit.PER_SECOND_AUDIO_OUT,
        }
    ),
    Benchmark.STT: frozenset(
        {
            PricingUnit.PER_MINUTE,
            PricingUnit.PER_HOUR,
            PricingUnit.PER_SECOND_AUDIO_IN,
        }
    ),
}


class PricingEntry(BaseModel, frozen=True, extra="forbid"):
    """One model's published rate, keyed like the model registry."""

    benchmark: Benchmark
    provider: str
    model: str
    unit: PricingUnit
    price_usd: Decimal = Field(gt=0)
    effective_from: date
    source_url: HttpUrl
    notes: str | None = None

    @field_validator("price_usd", mode="before")
    @classmethod
    def _price_written_as_string(cls, value: object) -> object:
        # A bare JSON number silently drops trailing zeros (0.20 → 0.2) — the
        # exact misquote the decimal-string convention exists to prevent.
        # Decimal itself stays accepted for entries built in code.
        if isinstance(value, float | int):
            raise ValueError('write price_usd as a string so the decimal is exact, e.g. "0.20"')
        return value

    @model_validator(mode="after")
    def _unit_bills_this_benchmark(self) -> PricingEntry:
        if self.unit not in _UNITS_BY_BENCHMARK.get(self.benchmark, frozenset()):
            raise ValueError(
                f"{self.unit} does not bill {self.benchmark} input; "
                "see the unit table in CONTRIBUTING.md"
            )
        return self

    @property
    def price_per_1m_chars(self) -> Decimal | None:
        """The native rate normalized to USD per 1M characters.

        Pure arithmetic per unit — ``per_1m_chars`` is served as-is,
        ``per_1k_chars`` scales by 1000, ``per_char`` by 1,000,000.
        ``per_second_audio_out`` returns None: seconds of audio have no
        character equivalence without assuming a speaking rate, and we never
        serve estimated figures.
        """
        if self.unit is PricingUnit.PER_1M_CHARS:
            return self.price_usd
        if self.unit is PricingUnit.PER_1K_CHARS:
            return self.price_usd * 1000
        if self.unit is PricingUnit.PER_CHAR:
            return self.price_usd * 1_000_000
        return None

    @property
    def price_per_1k_minutes(self) -> Decimal | None:
        """The native rate normalized to USD per 1,000 minutes of input audio.

        Defined only for the duration units that bill audio *in*.
        ``per_second_audio_out`` measures synthesized output against a
        different denominator, and character units have no duration
        equivalence without assuming a speaking rate — both return None
        rather than an estimate.

        Per-minute and per-second rates scale exactly; the hourly conversion
        divides and so rounds to ``_DIVIDED_SCALE``.
        """
        if self.unit is PricingUnit.PER_MINUTE:
            return self.price_usd * 1000
        if self.unit is PricingUnit.PER_HOUR:
            return (self.price_usd * 1000 / 60).quantize(_DIVIDED_SCALE, rounding=ROUND_HALF_UP)
        if self.unit is PricingUnit.PER_SECOND_AUDIO_IN:
            return self.price_usd * 60_000
        return None
