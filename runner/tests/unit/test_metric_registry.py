# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the metric registry."""

from __future__ import annotations

import pytest

from coval_bench.db.models import Benchmark
from coval_bench.registries import METRIC_SPECS, Metric, MetricDirection
from coval_bench.registries.metrics import METRIC_VALUE_CONTRACTS, MetricValueContract


def test_every_metric_has_a_spec() -> None:
    assert METRIC_SPECS.keys() == set(Metric)


def test_metric_values_match_stored_strings() -> None:
    # These are the exact strings prod writes to results.metric_type.
    assert {m.value for m in Metric} == {
        "WER",
        "TTFT",
        "TTFS",
        "TTFA",
        "TTFARoundtrip",
        "TTFALeadingSilence",
        "RTF",
        "AudioToFinal",
        "V2V",
        "InstructionFollowing",
        "InterruptionRate",
        "ExpectedBehaviorAdherence",
    }


def test_units_match_stored_strings() -> None:
    # These are the exact strings prod writes to results.metric_units.
    expected = {
        Metric.WER: "percent",
        Metric.TTFT: "seconds",
        Metric.TTFS: "seconds",
        Metric.TTFA: "milliseconds",
        Metric.TTFA_ROUNDTRIP: "milliseconds",
        Metric.TTFA_LEADING_SILENCE: "milliseconds",
        Metric.RTF: "ratio",
        Metric.AUDIO_TO_FINAL: "seconds",
        Metric.V2V: "milliseconds",
        Metric.INSTRUCTION_FOLLOWING: "percent",
        Metric.INTERRUPTION_RATE: "per_minute",
        Metric.EXPECTED_BEHAVIOR_ADHERENCE: "percent",
    }
    assert {m: spec.units for m, spec in METRIC_SPECS.items()} == expected


def test_benchmark_coverage() -> None:
    assert METRIC_SPECS[Metric.WER].benchmarks == {Benchmark.STT, Benchmark.TTS}
    for metric in (Metric.TTFA, Metric.TTFA_ROUNDTRIP, Metric.TTFA_LEADING_SILENCE):
        assert METRIC_SPECS[metric].benchmarks == {Benchmark.TTS}
    for metric in (Metric.TTFS, Metric.RTF, Metric.AUDIO_TO_FINAL):
        assert METRIC_SPECS[metric].benchmarks == {Benchmark.STT}
    assert METRIC_SPECS[Metric.TTFT].benchmarks == {Benchmark.STT, Benchmark.LLM}
    assert METRIC_SPECS[Metric.V2V].benchmarks == {Benchmark.S2S}
    assert METRIC_SPECS[Metric.INSTRUCTION_FOLLOWING].benchmarks == {Benchmark.S2S, Benchmark.LLM}
    assert METRIC_SPECS[Metric.EXPECTED_BEHAVIOR_ADHERENCE].benchmarks == {Benchmark.S2S}


def test_metric_directions() -> None:
    # Instruction adherence and expected behavior adherence are both pass
    # rates: higher is better. Every other metric is a latency/error measure:
    # lower is better.
    higher_is_better = {Metric.INSTRUCTION_FOLLOWING, Metric.EXPECTED_BEHAVIOR_ADHERENCE}
    assert all(
        METRIC_SPECS[m].direction is MetricDirection.HIGHER_IS_BETTER for m in higher_is_better
    )
    assert all(
        spec.direction is MetricDirection.LOWER_IS_BETTER
        for m, spec in METRIC_SPECS.items()
        if m not in higher_is_better
    )


def test_aggregation_declarations_are_explicit_and_versioned() -> None:
    assert MetricValueContract.model_fields["aggregation_method"].is_required()
    for (metric, version), rule in METRIC_VALUE_CONTRACTS.items():
        assert rule.metric == metric and rule.version == version
        assert rule.aggregation_method == ("ratio" if metric == Metric.WER else "mean")
    assert METRIC_VALUE_CONTRACTS[(Metric.WER, "v1")].ratio_fallback == "mean"


@pytest.mark.parametrize(
    "changes",
    [
        {"aggregation_method": "mean"},
        {"numerator_keys": ()},
        {"denominator_key": None},
        {"numerator_keys": ("unknown",)},
        {"numerator_keys": ("primary",)},
        {"numerator_keys": ("insertion_count", "insertion_count")},
        {"denominator_key": "insertion_count"},
        {"ratio_scale": 0},
        {"ratio_scale": -1},
        {"ratio_scale": float("inf")},
        {"ratio_scale": float("nan")},
    ],
)
def test_invalid_aggregation_contracts_fail_on_construction(changes: dict[str, object]) -> None:
    data = METRIC_VALUE_CONTRACTS[(Metric.WER, "v1")].model_dump()
    with pytest.raises(ValueError):
        MetricValueContract.model_validate({**data, **changes})
