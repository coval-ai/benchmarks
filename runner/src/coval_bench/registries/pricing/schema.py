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
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl

from coval_bench.registries.benchmarks import Benchmark


class PricingUnit(StrEnum):
    """The native unit a provider publishes its rate in.

    Character units bill TTS input, duration units bill STT input. Both
    normalize by exact arithmetic within their own denominator; nothing
    converts across the two, which would take an assumed speaking rate.
    """

    PER_1M_CHARS = "per_1m_chars"
    PER_1K_CHARS = "per_1k_chars"
    PER_CHAR = "per_char"
    PER_SECOND_AUDIO_OUT = "per_second_audio_out"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_SECOND_AUDIO_IN = "per_second_audio_in"


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
        """
        if self.unit is PricingUnit.PER_MINUTE:
            return self.price_usd * 1000
        if self.unit is PricingUnit.PER_HOUR:
            return self.price_usd * 1000 / 60
        if self.unit is PricingUnit.PER_SECOND_AUDIO_IN:
            return self.price_usd * 60_000
        return None
