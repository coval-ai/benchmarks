# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the S2S caller-condition metric contracts."""

from __future__ import annotations

from coval_bench.registries import Metric
from coval_bench.s2s import conditions


def test_standard_caller_requires_latency() -> None:
    standard = conditions.condition_for(conditions.DATASET_ID_MULTITURN)
    assert standard.required is Metric.V2V
    assert standard.optional == frozenset({Metric.INSTRUCTION_FOLLOWING, Metric.INTERRUPTION_RATE})


def test_noise_condition_excludes_latency() -> None:
    noisy = conditions.condition_for(conditions.DATASET_ID_MULTITURN_NOISY)
    assert noisy.required is Metric.INSTRUCTION_FOLLOWING
    assert noisy.optional == frozenset({Metric.INTERRUPTION_RATE})
    # Excluded by omission: never asked for, so its absence is never a warning.
    assert Metric.V2V not in noisy.fetched


def test_unmapped_dataset_keeps_the_pre_scoping_contract() -> None:
    legacy = conditions.condition_for(conditions.DATASET_ID)
    assert legacy == conditions.DEFAULT_CONDITION
    assert legacy.required is Metric.V2V


def test_every_fetched_metric_is_an_s2s_metric() -> None:
    from coval_bench.registries import METRIC_SPECS, Benchmark

    for dataset_id, condition in conditions.CONDITIONS.items():
        for metric in condition.fetched:
            assert Benchmark.S2S in METRIC_SPECS[metric].benchmarks, (dataset_id, metric)
