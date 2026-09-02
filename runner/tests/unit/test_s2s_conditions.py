# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the S2S caller-condition metric contracts."""

from __future__ import annotations

from coval_bench.registries import Benchmark, Metric
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


def test_llm_dental_contract_uses_instruction_and_local_ttft() -> None:
    llm = conditions.condition_for(conditions.DATASET_ID_LLM_DENTAL)
    assert llm.benchmark is Benchmark.LLM
    assert llm.required is Metric.INSTRUCTION_FOLLOWING
    assert llm.optional == frozenset()
    assert llm.local == frozenset({Metric.TTFT})
    assert (
        conditions.dataset_id_for(conditions.FAMILY_LLM_DENTAL, conditions.Condition.CLEAN)
        == conditions.DATASET_ID_LLM_DENTAL
    )
    assert (
        conditions.dataset_id_for(conditions.FAMILY_LLM_DENTAL, conditions.Condition.NOISY) is None
    )
    assert (
        conditions.dataset_id_for(conditions.FAMILY_LLM_DENTAL, conditions.Condition.ACCENTED)
        is None
    )
    existing = {
        dataset_id: condition
        for dataset_id, condition in conditions.CONDITIONS.items()
        if dataset_id != conditions.DATASET_ID_LLM_DENTAL
    }
    assert all(condition.benchmark is Benchmark.S2S for condition in existing.values())
    assert all(not condition.local for condition in existing.values())


def test_every_condition_metric_supports_its_benchmark() -> None:
    from coval_bench.registries import METRIC_SPECS

    for dataset_id, condition in conditions.CONDITIONS.items():
        for metric in condition.fetched | condition.local:
            assert condition.benchmark in METRIC_SPECS[metric].benchmarks, (dataset_id, metric)
