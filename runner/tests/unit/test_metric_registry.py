# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the metric registry."""

from __future__ import annotations

from coval_bench.db.models import Benchmark
from coval_bench.registries import METRIC_SPECS, Metric, MetricDirection
from coval_bench.registries.metrics import METRIC_VALUE_CONTRACTS


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
        "CallLength",
        "ExpectedBehaviorAdherence",
        "WorkflowAdherence",
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
        Metric.CALL_LENGTH: "seconds",
        Metric.EXPECTED_BEHAVIOR_ADHERENCE: "percent",
        Metric.WORKFLOW_ADHERENCE: "percent",
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
    assert METRIC_SPECS[Metric.WORKFLOW_ADHERENCE].benchmarks == {Benchmark.S2S}


def test_metric_directions() -> None:
    # The adherence metrics are pass rates: higher is better. Every other metric
    # is a latency/error measure: lower is better.
    higher_is_better = {
        Metric.INSTRUCTION_FOLLOWING,
        Metric.EXPECTED_BEHAVIOR_ADHERENCE,
        Metric.WORKFLOW_ADHERENCE,
    }
    assert all(
        METRIC_SPECS[m].direction is MetricDirection.HIGHER_IS_BETTER for m in higher_is_better
    )
    assert all(
        spec.direction is MetricDirection.LOWER_IS_BETTER
        for m, spec in METRIC_SPECS.items()
        if m not in higher_is_better
    )


def test_adherence_rates_are_capped_at_100_percent() -> None:
    for metric in (Metric.INSTRUCTION_FOLLOWING, Metric.WORKFLOW_ADHERENCE):
        (primary,) = METRIC_VALUE_CONTRACTS[(metric, "v1")].values
        assert (primary.minimum, primary.maximum) == (0.0, 100.0)
