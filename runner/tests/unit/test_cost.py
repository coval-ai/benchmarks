# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-item cost calculator and the judge-cost helper."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from coval_bench.db.models import Benchmark, BillingUnit, PriceRow, Result, ResultStatus
from coval_bench.metrics import compute_cost_usd, judge_cost_usd


def _rate(unit: BillingUnit, rate: str, benchmark: Benchmark = Benchmark.STT) -> PriceRow:
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


def _result(benchmark: Benchmark = Benchmark.STT, **usage: object) -> Result:
    return Result(
        run_id=1,
        provider="p",
        model="m",
        benchmark=benchmark,
        metric_type="WER",
        metric_value=1.0,
        metric_units="percent",
        status=ResultStatus.SUCCESS,
        **usage,  # type: ignore[arg-type]
    )


def test_token_rates_use_token_counts() -> None:
    rates = [
        _rate(BillingUnit.PER_1M_TOKENS_INPUT, "2.50"),
        _rate(BillingUnit.PER_1M_TOKENS_OUTPUT, "10.00"),
    ]
    result = _result(input_tokens=1_000_000, output_tokens=500_000)
    assert compute_cost_usd(result, rates) == pytest.approx(2.50 + 5.00)


def test_token_rates_partial_counts_still_price_available_side() -> None:
    rates = [
        _rate(BillingUnit.PER_1M_TOKENS_INPUT, "2.50"),
        _rate(BillingUnit.PER_1M_TOKENS_OUTPUT, "10.00"),
    ]
    result = _result(output_tokens=100_000)  # provider reported only output
    assert compute_cost_usd(result, rates) == pytest.approx(1.00)


def test_duration_rate_prefers_billable_seconds() -> None:
    rates = [_rate(BillingUnit.PER_MINUTE, "0.006")]
    result = _result(billable_seconds=120.0, audio_seconds_in=999.0)
    assert compute_cost_usd(result, rates) == pytest.approx(0.012)


def test_duration_rate_falls_back_to_audio_seconds_in_for_stt() -> None:
    rates = [_rate(BillingUnit.PER_HOUR, "0.36")]
    result = _result(audio_seconds_in=600.0)  # 1/6 hour
    assert compute_cost_usd(result, rates) == pytest.approx(0.06)


def test_duration_rate_falls_back_to_audio_seconds_out_for_tts() -> None:
    rates = [_rate(BillingUnit.PER_SECOND, "0.001", benchmark=Benchmark.TTS)]
    result = _result(benchmark=Benchmark.TTS, audio_seconds_out=30.0, audio_seconds_in=999.0)
    assert compute_cost_usd(result, rates) == pytest.approx(0.03)


def test_chars_rate_uses_characters_in() -> None:
    rates = [_rate(BillingUnit.PER_1M_CHARS, "50.00", benchmark=Benchmark.TTS)]
    result = _result(benchmark=Benchmark.TTS, characters_in=2_000)
    assert compute_cost_usd(result, rates) == pytest.approx(0.10)


def test_token_rates_without_counts_fall_through_to_duration() -> None:
    rates = [
        _rate(BillingUnit.PER_1M_TOKENS_INPUT, "2.50"),
        _rate(BillingUnit.PER_MINUTE, "0.006"),
    ]
    result = _result(billable_seconds=60.0)  # no tokens reported
    assert compute_cost_usd(result, rates) == pytest.approx(0.006)


def test_rates_from_other_benchmark_are_ignored() -> None:
    # gradium serves STT and TTS under one model id — the TTS chars rate must
    # not price an STT row.
    rates = [_rate(BillingUnit.PER_1M_CHARS, "48.00", benchmark=Benchmark.TTS)]
    result = _result(benchmark=Benchmark.STT, characters_in=1_000)
    assert compute_cost_usd(result, rates) is None


def test_no_rate_or_no_quantity_returns_none() -> None:
    assert compute_cost_usd(_result(billable_seconds=60.0), []) is None
    assert compute_cost_usd(_result(), [_rate(BillingUnit.PER_MINUTE, "0.006")]) is None
    assert compute_cost_usd(_result(), [_rate(BillingUnit.PER_REQUEST, "0.01")]) is None


def test_judge_cost_duration_usage() -> None:
    rates = [_rate(BillingUnit.PER_MINUTE, "0.006", benchmark=Benchmark.TTS)]
    assert judge_cost_usd({"audio_seconds": 300.0}, rates) == pytest.approx(0.03)


def test_judge_cost_token_usage() -> None:
    rates = [
        _rate(BillingUnit.PER_1M_TOKENS_INPUT, "2.00", benchmark=Benchmark.TTS),
        _rate(BillingUnit.PER_1M_TOKENS_OUTPUT, "4.00", benchmark=Benchmark.TTS),
    ]
    usage = {"input_tokens": 500_000.0, "output_tokens": 250_000.0}
    assert judge_cost_usd(usage, rates) == pytest.approx(1.00 + 1.00)


def test_judge_cost_empty_usage_or_rates_is_none() -> None:
    assert judge_cost_usd({}, [_rate(BillingUnit.PER_MINUTE, "0.006")]) is None
    assert judge_cost_usd({"audio_seconds": 300.0}, []) is None
