# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for every price-normalization branch."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from coval_bench.db.models import Benchmark, BillingUnit, PriceRow
from coval_bench.pricing.normalize import Conversion, normalized_price


def _rate(unit: BillingUnit, rate: str, benchmark: Benchmark) -> PriceRow:
    return PriceRow(
        provider="p",
        model="m",
        benchmark=benchmark,
        billing_unit=unit,
        rate_usd=Decimal(rate),
        effective_at=datetime(2026, 8, 1, tzinfo=UTC),
        source_url="https://example.com/pricing",
        as_of=date(2026, 8, 6),
        updated_by="human",
    )


def test_stt_per_minute() -> None:
    result = normalized_price(
        Benchmark.STT, [_rate(BillingUnit.PER_MINUTE, "0.006", Benchmark.STT)], None
    )
    assert result is not None
    assert result.value == pytest.approx(6.0)  # $/1k min
    assert result.basis == "list_price"


def test_stt_per_second() -> None:
    result = normalized_price(
        Benchmark.STT, [_rate(BillingUnit.PER_SECOND, "0.0001", Benchmark.STT)], None
    )
    assert result is not None
    assert result.value == pytest.approx(6.0)
    assert result.basis == "list_price"


def test_stt_per_hour() -> None:
    result = normalized_price(
        Benchmark.STT, [_rate(BillingUnit.PER_HOUR, "0.36", Benchmark.STT)], None
    )
    assert result is not None
    assert result.value == pytest.approx(6.0)
    assert result.basis == "list_price"


def test_stt_token_rates_with_measured_conversion() -> None:
    rates = [
        _rate(BillingUnit.PER_1M_TOKENS_INPUT, "2.50", Benchmark.STT),
        _rate(BillingUnit.PER_1M_TOKENS_OUTPUT, "10.00", Benchmark.STT),
    ]
    conversion = Conversion(in_tokens_per_min=600.0, out_tokens_per_min=100.0, sample_count=100)
    result = normalized_price(Benchmark.STT, rates, conversion)
    assert result is not None
    # (2.5/1e6*600 + 10/1e6*100) * 1000 = 1.5 + 1.0
    assert result.value == pytest.approx(2.5)
    assert result.basis == "list_price_measured_conversion"


def test_stt_token_rates_without_conversion_is_none() -> None:
    rates = [_rate(BillingUnit.PER_1M_TOKENS_INPUT, "2.50", Benchmark.STT)]
    assert normalized_price(Benchmark.STT, rates, None) is None
    assert normalized_price(Benchmark.STT, rates, Conversion(sample_count=100)) is None


def test_tts_per_1m_chars_passthrough() -> None:
    result = normalized_price(
        Benchmark.TTS, [_rate(BillingUnit.PER_1M_CHARS, "50.00", Benchmark.TTS)], None
    )
    assert result is not None
    assert result.value == pytest.approx(50.0)
    assert result.basis == "list_price"


def test_tts_per_second_with_measured_chars_per_sec() -> None:
    conversion = Conversion(chars_per_sec=16.0, sample_count=100)
    result = normalized_price(
        Benchmark.TTS, [_rate(BillingUnit.PER_SECOND, "0.001", Benchmark.TTS)], conversion
    )
    assert result is not None
    # $0.001/s of audio; 1M chars at 16 chars/s of audio = 62,500s → $62.50
    assert result.value == pytest.approx(62.5)
    assert result.basis == "list_price_measured_conversion"


def test_tts_per_second_without_conversion_is_none() -> None:
    rates = [_rate(BillingUnit.PER_SECOND, "0.001", Benchmark.TTS)]
    assert normalized_price(Benchmark.TTS, rates, None) is None


def test_tts_token_rates_with_measured_tokens_per_char() -> None:
    rates = [
        _rate(BillingUnit.PER_1M_TOKENS_INPUT, "0.60", Benchmark.TTS),
        _rate(BillingUnit.PER_1M_TOKENS_OUTPUT, "12.00", Benchmark.TTS),
    ]
    conversion = Conversion(in_tokens_per_char=0.25, out_tokens_per_char=5.0, sample_count=100)
    result = normalized_price(Benchmark.TTS, rates, conversion)
    assert result is not None
    # per char: 0.6/1e6*0.25 + 12/1e6*5 → ×1e6 chars = 0.15 + 60
    assert result.value == pytest.approx(60.15)
    assert result.basis == "list_price_measured_conversion"


def test_tts_token_rates_without_conversion_is_none() -> None:
    rates = [_rate(BillingUnit.PER_1M_TOKENS_OUTPUT, "12.00", Benchmark.TTS)]
    assert normalized_price(Benchmark.TTS, rates, None) is None


def test_s2s_per_minute() -> None:
    result = normalized_price(
        Benchmark.S2S, [_rate(BillingUnit.PER_MINUTE, "0.05", Benchmark.S2S)], None
    )
    assert result is not None
    assert result.value == pytest.approx(50.0)
    assert result.basis == "list_price"


def test_rates_from_other_benchmark_ignored() -> None:
    # gradium serves STT and TTS under one model id.
    rates = [_rate(BillingUnit.PER_1M_CHARS, "48.00", Benchmark.TTS)]
    assert normalized_price(Benchmark.STT, rates, None) is None


def test_per_request_rate_never_normalized() -> None:
    rates = [_rate(BillingUnit.PER_REQUEST, "0.01", Benchmark.STT)]
    assert normalized_price(Benchmark.STT, rates, None) is None
