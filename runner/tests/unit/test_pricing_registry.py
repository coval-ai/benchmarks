# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pricing registry: schema validation, normalization, and MODEL_REGISTRY parity."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import MODEL_REGISTRY
from coval_bench.registries.pricing import PRICING, PricingEntry, PricingUnit, index_pricing


def _entry(**overrides: object) -> PricingEntry:
    fields: dict[str, object] = {
        "benchmark": Benchmark.TTS,
        "provider": "deepgram",
        "model": "aura-2-thalia-en",
        "unit": PricingUnit.PER_1K_CHARS,
        "price_usd": Decimal("0.030"),
        "effective_from": date(2026, 8, 10),
        "source_url": "https://deepgram.com/pricing",
    }
    fields.update(overrides)
    return PricingEntry.model_validate(fields)


def test_shipped_registry_matches_model_registry() -> None:
    """Every shipped entry keys to a registered model; both benchmarks are priced."""
    assert PRICING
    registered = {(m.benchmark, m.provider, m.model) for m in MODEL_REGISTRY}
    assert set(PRICING) <= registered
    assert {k[0] for k in PRICING} == {Benchmark.TTS, Benchmark.STT}


def test_every_shipped_entry_serves_its_benchmark_figure() -> None:
    """Each STT rate normalizes to $/1k minutes; TTS rates that can, to $/1M chars.

    ``per_second_audio_out`` is the one TTS unit allowed to serve nothing —
    output seconds have no character equivalence without a speaking rate.
    """
    for (benchmark, _, _), entry in PRICING.items():
        if benchmark is Benchmark.STT:
            assert entry.price_per_1k_minutes is not None
            assert entry.price_per_1m_chars is None
        elif entry.unit is not PricingUnit.PER_SECOND_AUDIO_OUT:
            assert entry.price_per_1m_chars is not None


@pytest.mark.parametrize(
    ("unit", "price", "normalized"),
    [
        (PricingUnit.PER_1M_CHARS, Decimal("15"), Decimal("15")),
        (PricingUnit.PER_1K_CHARS, Decimal("0.030"), Decimal("30")),
        (PricingUnit.PER_CHAR, Decimal("0.00003"), Decimal("30")),
        (PricingUnit.PER_SECOND_AUDIO_OUT, Decimal("0.001"), None),
    ],
)
def test_normalization(unit: PricingUnit, price: Decimal, normalized: Decimal | None) -> None:
    """Native units normalize to $/1M chars; audio-out seconds never estimate one."""
    assert _entry(unit=unit, price_usd=price).price_per_1m_chars == normalized


@pytest.mark.parametrize(
    ("unit", "price", "normalized"),
    [
        (PricingUnit.PER_MINUTE, Decimal("0.0043"), Decimal("4.3")),
        (PricingUnit.PER_HOUR, Decimal("0.36"), Decimal("6")),
        # The hourly conversion divides, so a rate that does not terminate
        # lands on a fixed scale rather than the Decimal context's tail.
        (PricingUnit.PER_HOUR, Decimal("0.43"), Decimal("7.166667")),
        (PricingUnit.PER_SECOND_AUDIO_IN, Decimal("0.0001"), Decimal("6")),
        (PricingUnit.PER_1K_CHARS, Decimal("0.030"), None),
        (PricingUnit.PER_SECOND_AUDIO_OUT, Decimal("0.001"), None),
    ],
)
def test_minutes_normalization(
    unit: PricingUnit, price: Decimal, normalized: Decimal | None
) -> None:
    """Duration units normalize to $/1k minutes; character units never estimate one."""
    stt = unit in (PricingUnit.PER_MINUTE, PricingUnit.PER_HOUR, PricingUnit.PER_SECOND_AUDIO_IN)
    entry = _entry(unit=unit, price_usd=price, benchmark=Benchmark.STT if stt else Benchmark.TTS)
    assert entry.price_per_1k_minutes == normalized


def test_duplicate_key_raises() -> None:
    entry = _entry()
    with pytest.raises(RuntimeError, match="duplicate"):
        index_pricing([entry, _entry(notes="same key, different rate")])
    assert index_pricing([entry]) == {(entry.benchmark, entry.provider, entry.model): entry}


def test_unregistered_model_raises() -> None:
    with pytest.raises(RuntimeError, match="no MODEL_REGISTRY row"):
        index_pricing([_entry(model="aura-99")])


@pytest.mark.parametrize(
    "overrides",
    [
        {"price_usd": Decimal("0")},
        {"price_usd": Decimal("-1")},
        # A bare JSON number silently drops trailing zeros (0.20 → 0.2) — the
        # misquote the decimal-string convention exists to prevent.
        {"price_usd": 0.20},
        {"price_usd": 1},
        {"source_url": "not a url"},
        {"unit": "per_word"},
        # A duration unit cannot bill TTS input, nor a character unit STT.
        {"benchmark": Benchmark.STT},
        {"unit": PricingUnit.PER_HOUR},
        {"currency": "EUR"},
    ],
)
def test_schema_rejects_invalid_entries(overrides: dict[str, object]) -> None:
    """Zero/negative/numeric rates, bad URLs, unit-benchmark mismatches, and stray fields fail."""
    with pytest.raises(ValidationError):
        _entry(**overrides)


def test_no_shipped_rate_is_future_dated() -> None:
    """The registry holds one entry per model, so a future-dated entry replaces
    the rate still in force and the model serves no price until the date
    arrives. Date entries at or before the day the PR can merge."""
    future = sorted(
        f"{b}:{p}/{m}"
        for (b, p, m), entry in PRICING.items()
        if entry.effective_from > date.today()
    )
    assert future == [], f"ratesheets carry future-dated rates: {', '.join(future)}"
