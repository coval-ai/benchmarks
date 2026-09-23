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


def test_bank_board_anchors_on_latency_and_carries_the_judge() -> None:
    """Latency is on every Ultra Bank run; the expected-behaviors judge rides along
    under the instruction metric so the adherence chart needs no second series."""
    bank = conditions.condition_for(conditions.DATASET_ID_BANK)
    assert bank.required is Metric.V2V
    assert bank.optional == frozenset(
        {Metric.INSTRUCTION_FOLLOWING, Metric.INTERRUPTION_RATE, Metric.CALL_LENGTH}
    )
    assert (
        conditions.dataset_id_for(conditions.FAMILY_BANK, conditions.Condition.CLEAN)
        == conditions.DATASET_ID_BANK
    )
    # The generic noise personas never run the bank set; its noise comes as tiers.
    assert conditions.dataset_id_for(conditions.FAMILY_BANK, conditions.Condition.NOISY) is None
    assert conditions.dataset_id_for(conditions.FAMILY_BANK, conditions.Condition.ACCENTED) is None


def test_bank_tiers_anchor_on_the_judge_and_never_ask_for_latency() -> None:
    """Each difficulty tier is its own dataset, so noise never pools into the
    clean board, and none of them fetches V2V."""
    tiers = {
        conditions.Condition.LOW: conditions.DATASET_ID_BANK_LOW,
        conditions.Condition.MEDIUM: conditions.DATASET_ID_BANK_MEDIUM,
        conditions.Condition.HARD: conditions.DATASET_ID_BANK_HARD,
        conditions.Condition.EXTRA_HARD: conditions.DATASET_ID_BANK_EXTRA_HARD,
    }
    assert len(set(tiers.values()) | {conditions.DATASET_ID_BANK}) == 5
    for condition, dataset_id in tiers.items():
        assert conditions.dataset_id_for(conditions.FAMILY_BANK, condition) == dataset_id
        contract = conditions.condition_for(dataset_id)
        assert contract.required is Metric.INSTRUCTION_FOLLOWING
        assert contract.optional == frozenset({Metric.INTERRUPTION_RATE, Metric.CALL_LENGTH})
        assert Metric.V2V not in contract.fetched


def test_unmapped_dataset_keeps_the_pre_scoping_contract() -> None:
    legacy = conditions.condition_for(conditions.DATASET_ID)
    assert legacy == conditions.DEFAULT_CONDITION
    assert legacy.required is Metric.V2V


def test_llm_contracts_use_instruction_and_local_ttft() -> None:
    llm_families = {
        conditions.FAMILY_LLM_DENTAL: conditions.DATASET_ID_LLM_DENTAL,
        conditions.FAMILY_LLM_BANK: conditions.DATASET_ID_LLM_BANK,
    }
    for family, dataset_id in llm_families.items():
        llm = conditions.condition_for(dataset_id)
        assert llm.benchmark is Benchmark.LLM
        assert llm.required is Metric.INSTRUCTION_FOLLOWING
        assert llm.optional == frozenset()
        assert llm.local == frozenset({Metric.TTFT})
        assert conditions.dataset_id_for(family, conditions.Condition.CLEAN) == dataset_id
        assert conditions.dataset_id_for(family, conditions.Condition.NOISY) is None
    voice = {
        dataset_id: condition
        for dataset_id, condition in conditions.CONDITIONS.items()
        if dataset_id not in llm_families.values()
    }
    assert all(condition.benchmark is Benchmark.S2S for condition in voice.values())
    assert all(not condition.local for condition in voice.values())


def test_every_condition_metric_supports_its_benchmark() -> None:
    from coval_bench.registries import METRIC_SPECS

    for dataset_id, condition in conditions.CONDITIONS.items():
        for metric in condition.fetched | condition.local:
            assert condition.benchmark in METRIC_SPECS[metric].benchmarks, (dataset_id, metric)


def test_scenario_dataset_ids_derive_from_the_slug() -> None:
    """Bank keeps the ids the board already stores; new domains follow the same shape."""
    assert conditions.scenario_family("bank") == conditions.FAMILY_BANK
    assert conditions.scenario_dataset_id("bank", conditions.Condition.CLEAN) == "s2s-bank-v1"
    assert (
        conditions.scenario_dataset_id("bank", conditions.Condition.EXTRA_HARD)
        == conditions.DATASET_ID_BANK_EXTRA_HARD
    )
    assert conditions.scenario_dataset_id("happy-smile", conditions.Condition.HARD) == (
        "s2s-happy-smile-hard-v1"
    )
    assert conditions.scenario_dataset_id("happy-customer", conditions.Condition.NOISY) is None
    for slug in conditions.SCENARIO_SLUGS:
        family = conditions.scenario_family(slug)
        clean = conditions.dataset_id_for(family, conditions.Condition.CLEAN)
        assert clean is not None
        assert conditions.condition_for(clean).required is Metric.V2V
        for tier in conditions.TIERS:
            tier_id = conditions.dataset_id_for(family, tier)
            assert tier_id is not None
            assert conditions.condition_for(tier_id).required is Metric.INSTRUCTION_FOLLOWING
