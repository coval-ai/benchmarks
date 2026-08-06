# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-item spend at the effective list rate, in USD.

Resolution order mirrors how providers actually bill: token rates first (the
only exact signal), then billed duration (provider-reported
``billable_seconds``, falling back to the measured audio duration), then
characters. No resolvable rate or no quantity → ``None`` — spend is never
guessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coval_bench.db.models import Benchmark, BillingUnit

if TYPE_CHECKING:
    from coval_bench.db.models import PriceRow, Result

_SECONDS_PER_UNIT: dict[BillingUnit, float] = {
    BillingUnit.PER_SECOND: 1.0,
    BillingUnit.PER_MINUTE: 60.0,
    BillingUnit.PER_HOUR: 3600.0,
}


def compute_cost_usd(result: Result, rates: list[PriceRow]) -> float | None:
    """USD cost of one benchmark item given its usage and the effective rates.

    *result* is any metric row of the item — usage quantities are stamped
    identically on all of them. Rates for other benchmarks (gradium serves STT
    and TTS under one model id) are ignored.
    """
    by_unit = {r.billing_unit: r for r in rates if r.benchmark is result.benchmark}

    in_rate = by_unit.get(BillingUnit.PER_1M_TOKENS_INPUT)
    out_rate = by_unit.get(BillingUnit.PER_1M_TOKENS_OUTPUT)
    if in_rate is not None or out_rate is not None:
        cost: float | None = None
        if in_rate is not None and result.input_tokens is not None:
            cost = (cost or 0.0) + result.input_tokens / 1e6 * float(in_rate.rate_usd)
        if out_rate is not None and result.output_tokens is not None:
            cost = (cost or 0.0) + result.output_tokens / 1e6 * float(out_rate.rate_usd)
        if cost is not None:
            return cost

    for unit, seconds_per in _SECONDS_PER_UNIT.items():
        rate = by_unit.get(unit)
        if rate is None:
            continue
        seconds = result.billable_seconds
        if seconds is None:
            seconds = (
                result.audio_seconds_out
                if result.benchmark is Benchmark.TTS
                else result.audio_seconds_in
            )
        if seconds is not None:
            return seconds / seconds_per * float(rate.rate_usd)

    chars_rate = by_unit.get(BillingUnit.PER_1M_CHARS)
    if chars_rate is not None and result.characters_in is not None:
        return result.characters_in / 1e6 * float(chars_rate.rate_usd)

    return None


def judge_cost_usd(judge_usage: dict[str, float], rates: list[PriceRow]) -> float | None:
    """USD cost of the run's whisper-1 judge calls from the accumulated usage.

    Handles both usage shapes the judge can report: duration seconds against a
    per-duration rate, token counts against token rates.
    """
    by_unit = {r.billing_unit: r for r in rates}
    cost: float | None = None
    seconds = judge_usage.get("audio_seconds")
    if seconds is not None:
        for unit, seconds_per in _SECONDS_PER_UNIT.items():
            rate = by_unit.get(unit)
            if rate is not None:
                cost = seconds / seconds_per * float(rate.rate_usd)
                break
    for key, unit in (
        ("input_tokens", BillingUnit.PER_1M_TOKENS_INPUT),
        ("output_tokens", BillingUnit.PER_1M_TOKENS_OUTPUT),
    ):
        tokens = judge_usage.get(key)
        rate = by_unit.get(unit)
        if tokens is not None and rate is not None:
            cost = (cost or 0.0) + tokens / 1e6 * float(rate.rate_usd)
    return cost
