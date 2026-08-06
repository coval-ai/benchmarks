# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Normalize native billing rates to display units, AA-style.

Targets: USD per 1,000 minutes of audio (STT, S2S) and USD per 1M characters
(TTS). Duration- and character-billed rates convert arithmetically
(``basis="list_price"``). Token- and per-second-billed models convert through
conversion rates measured from our own runs over the last 7 days
(``basis="list_price_measured_conversion"``) — the same idea Artificial
Analysis applies to token billing, but re-measured continuously instead of
one-off. No measured conversion (or too few samples) → no normalized value,
never an estimate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from coval_bench.db.models import Benchmark, BillingUnit

if TYPE_CHECKING:
    import psycopg
    import psycopg.rows
    from psycopg_pool import AsyncConnectionPool

    from coval_bench.db.models import PriceRow

#: Minimum anchor rows in the window before a measured conversion is trusted.
MIN_CONVERSION_SAMPLES = 50
CONVERSION_WINDOW = "7d"

_SECONDS_PER_UNIT: dict[BillingUnit, float] = {
    BillingUnit.PER_SECOND: 1.0,
    BillingUnit.PER_MINUTE: 60.0,
    BillingUnit.PER_HOUR: 3600.0,
}

# One anchor metric per benchmark (AudioToFinal / TTFA / V2V): exactly one row
# per item carries the item's usage quantities, so summing over anchors never
# double-counts the usage columns duplicated across an item's metric rows.
# Each ratio is computed strictly over rows where BOTH its numerator and
# denominator are present (FILTER pairs), and each carries its own sample
# count — COUNT(*) over the anchor rows would overstate the population when
# providers report usage on only some rows, biasing the published conversion.
_CONVERSIONS_SQL = """
    SELECT provider, model, benchmark,
           COUNT(*) FILTER (WHERE input_tokens IS NOT NULL
                              AND audio_seconds_in > 0)::int AS in_tpm_n,
           SUM(input_tokens)     FILTER (WHERE input_tokens IS NOT NULL
                              AND audio_seconds_in > 0)::float8 AS in_tpm_tokens,
           SUM(audio_seconds_in) FILTER (WHERE input_tokens IS NOT NULL
                              AND audio_seconds_in > 0)::float8 AS in_tpm_seconds,
           COUNT(*) FILTER (WHERE output_tokens IS NOT NULL
                              AND audio_seconds_in > 0)::int AS out_tpm_n,
           SUM(output_tokens)    FILTER (WHERE output_tokens IS NOT NULL
                              AND audio_seconds_in > 0)::float8 AS out_tpm_tokens,
           SUM(audio_seconds_in) FILTER (WHERE output_tokens IS NOT NULL
                              AND audio_seconds_in > 0)::float8 AS out_tpm_seconds,
           COUNT(*) FILTER (WHERE characters_in IS NOT NULL
                              AND audio_seconds_out > 0)::int AS cps_n,
           SUM(characters_in)     FILTER (WHERE characters_in IS NOT NULL
                              AND audio_seconds_out > 0)::float8 AS cps_chars,
           SUM(audio_seconds_out) FILTER (WHERE characters_in IS NOT NULL
                              AND audio_seconds_out > 0)::float8 AS cps_seconds,
           COUNT(*) FILTER (WHERE input_tokens IS NOT NULL
                              AND characters_in > 0)::int AS in_tpc_n,
           SUM(input_tokens)  FILTER (WHERE input_tokens IS NOT NULL
                              AND characters_in > 0)::float8 AS in_tpc_tokens,
           SUM(characters_in) FILTER (WHERE input_tokens IS NOT NULL
                              AND characters_in > 0)::float8 AS in_tpc_chars,
           COUNT(*) FILTER (WHERE output_tokens IS NOT NULL
                              AND characters_in > 0)::int AS out_tpc_n,
           SUM(output_tokens) FILTER (WHERE output_tokens IS NOT NULL
                              AND characters_in > 0)::float8 AS out_tpc_tokens,
           SUM(characters_in) FILTER (WHERE output_tokens IS NOT NULL
                              AND characters_in > 0)::float8 AS out_tpc_chars
    FROM benchmarks_v2.results
    WHERE status = 'success'
      AND created_at >= now() - INTERVAL '7 days'
      AND ((benchmark = 'STT' AND metric_type = 'AudioToFinal')
        OR (benchmark = 'TTS' AND metric_type = 'TTFA')
        OR (benchmark = 'S2S' AND metric_type = 'V2V'))
    GROUP BY provider, model, benchmark
"""


class Conversion(BaseModel):
    """Measured unit-conversion rates for one (provider, model, benchmark)."""

    in_tokens_per_min: float | None = None
    out_tokens_per_min: float | None = None
    chars_per_sec: float | None = None
    in_tokens_per_char: float | None = None
    out_tokens_per_char: float | None = None
    sample_count: int
    window: str = CONVERSION_WINDOW


class NormalizedPrice(BaseModel):
    """A display price in the benchmark's target unit, with its derivation basis."""

    value: float
    basis: Literal["list_price", "list_price_measured_conversion"]


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


async def load_conversions(
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]],
) -> dict[tuple[str, str, Benchmark], Conversion]:
    """Measured conversion rates for every model with enough recent samples.

    One grouped query over the 7-day window; models under
    ``MIN_CONVERSION_SAMPLES`` anchor rows are omitted entirely.
    """
    import psycopg.rows as _rows

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=_rows.dict_row) as cur:
            await cur.execute(_CONVERSIONS_SQL)
            rows = await cur.fetchall()
        await conn.commit()

    conversions: dict[tuple[str, str, Benchmark], Conversion] = {}
    for r in rows:

        def _pair(n_key: str, num_key: str, den_key: str, scale: float = 1.0) -> float | None:
            # A ratio exists only when its own paired population clears the
            # floor; each pair's numerator and denominator come from the same
            # rows by construction of the FILTER clauses.
            if r[n_key] < MIN_CONVERSION_SAMPLES:  # noqa: B023 — consumed within the iteration
                return None
            return _ratio(r[num_key], (r[den_key] or 0) / scale)  # noqa: B023

        fields = {
            "in_tokens_per_min": _pair("in_tpm_n", "in_tpm_tokens", "in_tpm_seconds", 60),
            "out_tokens_per_min": _pair("out_tpm_n", "out_tpm_tokens", "out_tpm_seconds", 60),
            "chars_per_sec": _pair("cps_n", "cps_chars", "cps_seconds"),
            "in_tokens_per_char": _pair("in_tpc_n", "in_tpc_tokens", "in_tpc_chars"),
            "out_tokens_per_char": _pair("out_tpc_n", "out_tpc_tokens", "out_tpc_chars"),
        }
        counts = {
            "in_tokens_per_min": r["in_tpm_n"],
            "out_tokens_per_min": r["out_tpm_n"],
            "chars_per_sec": r["cps_n"],
            "in_tokens_per_char": r["in_tpc_n"],
            "out_tokens_per_char": r["out_tpc_n"],
        }
        present = {k: v for k, v in fields.items() if v is not None}
        if not present:
            continue
        conversions[(r["provider"], r["model"], Benchmark(r["benchmark"]))] = Conversion(
            **fields,
            # The smallest population among the ratios actually served — the
            # conservative honest answer to "how many samples back this".
            sample_count=min(counts[k] for k in present),
        )
    return conversions


def normalized_price(
    benchmark: Benchmark,
    rates: list[PriceRow],
    conversion: Conversion | None,
) -> NormalizedPrice | None:
    """Normalize one model's effective rates to the benchmark's display unit.

    STT/S2S → USD per 1,000 minutes; TTS → USD per 1M characters. Returns
    ``None`` when the native unit needs a measured conversion that isn't
    available — the caller shows native rates with a null normalized value.
    """
    by_unit = {r.billing_unit: r for r in rates if r.benchmark is benchmark}
    duration_rate = next(((u, r) for u, r in by_unit.items() if u in _SECONDS_PER_UNIT), None)
    in_rate = by_unit.get(BillingUnit.PER_1M_TOKENS_INPUT)
    out_rate = by_unit.get(BillingUnit.PER_1M_TOKENS_OUTPUT)

    if benchmark is Benchmark.TTS:
        chars_rate = by_unit.get(BillingUnit.PER_1M_CHARS)
        if chars_rate is not None:
            return NormalizedPrice(value=float(chars_rate.rate_usd), basis="list_price")
        if duration_rate is not None:
            unit, rate = duration_rate
            chars_per_sec = conversion.chars_per_sec if conversion else None
            if chars_per_sec is None:
                return None
            per_second = float(rate.rate_usd) / _SECONDS_PER_UNIT[unit]
            return NormalizedPrice(
                value=per_second * (1e6 / chars_per_sec),
                basis="list_price_measured_conversion",
            )
        if in_rate is not None or out_rate is not None:
            in_tpc = conversion.in_tokens_per_char if conversion else None
            out_tpc = conversion.out_tokens_per_char if conversion else None
            if (in_rate is not None and in_tpc is None) or (
                out_rate is not None and out_tpc is None
            ):
                return None
            value = 0.0
            if in_rate is not None and in_tpc is not None:
                value += float(in_rate.rate_usd) / 1e6 * in_tpc * 1e6
            if out_rate is not None and out_tpc is not None:
                value += float(out_rate.rate_usd) / 1e6 * out_tpc * 1e6
            return NormalizedPrice(value=value, basis="list_price_measured_conversion")
        return None

    # STT / S2S → USD per 1,000 minutes
    if duration_rate is not None:
        unit, rate = duration_rate
        per_minute = float(rate.rate_usd) * 60 / _SECONDS_PER_UNIT[unit]
        return NormalizedPrice(value=per_minute * 1000, basis="list_price")
    if in_rate is not None or out_rate is not None:
        in_tpm = conversion.in_tokens_per_min if conversion else None
        out_tpm = conversion.out_tokens_per_min if conversion else None
        if (in_rate is not None and in_tpm is None) or (out_rate is not None and out_tpm is None):
            return None
        per_minute = 0.0
        if in_rate is not None and in_tpm is not None:
            per_minute += float(in_rate.rate_usd) / 1e6 * in_tpm
        if out_rate is not None and out_tpm is not None:
            per_minute += float(out_rate.rate_usd) / 1e6 * out_tpm
        return NormalizedPrice(value=per_minute * 1000, basis="list_price_measured_conversion")
    return None
