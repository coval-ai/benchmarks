# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the metric registry."""

from __future__ import annotations

from coval_bench.db.models import Benchmark
from coval_bench.registries import METRIC_SPECS, Metric, MetricDirection


def test_every_metric_has_a_spec() -> None:
    assert METRIC_SPECS.keys() == set(Metric)


def test_metric_values_match_stored_strings() -> None:
    # These are the exact strings prod writes to results.metric_type.
    assert {m.value for m in Metric} == {
        "WER",
        "TTFT",
        "TTFS",
        "TTFA",
        "RTF",
        "AudioToFinal",
    }


def test_units_match_stored_strings() -> None:
    # These are the exact strings prod writes to results.metric_units.
    expected = {
        Metric.WER: "percent",
        Metric.TTFT: "seconds",
        Metric.TTFS: "seconds",
        Metric.TTFA: "milliseconds",
        Metric.RTF: "ratio",
        Metric.AUDIO_TO_FINAL: "seconds",
    }
    assert {m: spec.units for m, spec in METRIC_SPECS.items()} == expected


def test_benchmark_coverage() -> None:
    assert METRIC_SPECS[Metric.WER].benchmarks == {Benchmark.STT, Benchmark.TTS}
    assert METRIC_SPECS[Metric.TTFA].benchmarks == {Benchmark.TTS}
    for metric in (Metric.TTFT, Metric.TTFS, Metric.RTF, Metric.AUDIO_TO_FINAL):
        assert METRIC_SPECS[metric].benchmarks == {Benchmark.STT}


def test_all_current_metrics_are_lower_is_better() -> None:
    assert all(
        spec.direction is MetricDirection.LOWER_IS_BETTER for spec in METRIC_SPECS.values()
    )
