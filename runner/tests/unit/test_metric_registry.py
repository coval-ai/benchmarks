# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the metric registry."""

from __future__ import annotations

from coval_bench.db.models import Benchmark
from coval_bench.registries import METRIC_SPECS, Metric, MetricDirection
from coval_bench.registries.metrics import METRIC_VALUE_CONTRACTS, headline_metrics


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
        "CharacterErrorRate",
        "SemDistMiniLM",
        "ToolCallCount",
        "InterruptionCueLanded",
        "InterruptionYielded",
        "BackchannelContinued",
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
        Metric.CHARACTER_ERROR_RATE: "percent",
        Metric.SEMDIST_MINILM: "ratio",
        Metric.TOOL_CALL_COUNT: "count",
        Metric.INTERRUPTION_CUE_LANDED: "count",
        Metric.INTERRUPTION_YIELDED: "percent",
        Metric.BACKCHANNEL_CONTINUED: "percent",
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
    assert METRIC_SPECS[Metric.CHARACTER_ERROR_RATE].benchmarks == {
        Benchmark.STT,
        Benchmark.TTS,
        Benchmark.S2S,
    }
    assert METRIC_SPECS[Metric.TOOL_CALL_COUNT].benchmarks == {Benchmark.S2S}
    for metric in (
        Metric.INTERRUPTION_CUE_LANDED,
        Metric.INTERRUPTION_YIELDED,
        Metric.BACKCHANNEL_CONTINUED,
    ):
        assert METRIC_SPECS[metric].benchmarks == {Benchmark.S2S}


def test_metric_directions() -> None:
    # The adherence and interruption judges are pass rates: higher is better. Tool
    # Call Count has a right answer per test case and Cue Landed is a validity gate,
    # so neither has a direction. Every other metric is a latency/error measure:
    # lower is better.
    undirected = {Metric.TOOL_CALL_COUNT, Metric.INTERRUPTION_CUE_LANDED}
    assert all(METRIC_SPECS[m].direction is None for m in undirected)
    higher_is_better = {
        Metric.INSTRUCTION_FOLLOWING,
        Metric.EXPECTED_BEHAVIOR_ADHERENCE,
        Metric.WORKFLOW_ADHERENCE,
        Metric.INTERRUPTION_YIELDED,
        Metric.BACKCHANNEL_CONTINUED,
    }
    assert all(
        METRIC_SPECS[m].direction is MetricDirection.HIGHER_IS_BETTER for m in higher_is_better
    )
    assert all(
        spec.direction is MetricDirection.LOWER_IS_BETTER
        for m, spec in METRIC_SPECS.items()
        if m not in higher_is_better and m not in undirected
    )


def test_adherence_rates_are_capped_at_100_percent() -> None:
    for metric in (
        Metric.INSTRUCTION_FOLLOWING,
        Metric.WORKFLOW_ADHERENCE,
        Metric.INTERRUPTION_YIELDED,
        Metric.BACKCHANNEL_CONTINUED,
    ):
        (primary,) = METRIC_VALUE_CONTRACTS[(metric, "v1")].values
        assert (primary.minimum, primary.maximum) == (0.0, 100.0)


def test_headline_metrics_come_from_the_specs() -> None:
    assert set(headline_metrics(Benchmark.STT)) == {
        Metric.WER,
        Metric.TTFT,
        Metric.TTFS,
        Metric.RTF,
        Metric.AUDIO_TO_FINAL,
        Metric.CHARACTER_ERROR_RATE,
        Metric.SEMDIST_MINILM,
    }
    assert set(headline_metrics(Benchmark.TTS)) == {
        Metric.WER,
        Metric.TTFA,
        Metric.CHARACTER_ERROR_RATE,
        Metric.SEMDIST_MINILM,
    }
    assert {m for m, spec in METRIC_SPECS.items() if spec.scored_from_text} == {
        Metric.WER,
        Metric.CHARACTER_ERROR_RATE,
        Metric.SEMDIST_MINILM,
    }
