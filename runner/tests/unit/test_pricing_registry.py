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
    """Every shipped entry keys to a registered model; the registry is non-empty."""
    assert PRICING
    registered = {(m.benchmark, m.provider, m.model) for m in MODEL_REGISTRY}
    assert set(PRICING) <= registered
    assert all(k[0] is Benchmark.TTS for k in PRICING)


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
        {"source_url": "not a url"},
        {"unit": "per_word"},
        {"currency": "EUR"},
    ],
)
def test_schema_rejects_invalid_entries(overrides: dict[str, object]) -> None:
    """Zero/negative rates, bad URLs, unknown units, and stray fields all fail validation."""
    with pytest.raises(ValidationError):
        _entry(**overrides)
