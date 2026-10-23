# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The shared scenario resolves the same ids for both modalities."""

from __future__ import annotations

import pytest

from coval_bench import scenarios
from coval_bench.config import Settings
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.metrics import Metric
from coval_bench.s2s.conditions import (
    CONDITIONS,
    DATASET_ID_BANK,
    DATASET_ID_LLM_BANK,
    SCENARIO_SLUGS,
)


def test_bank_scenario_names_one_dataset_per_benchmark() -> None:
    assert scenarios.ACTIVE.primary_dataset(Benchmark.S2S) == DATASET_ID_BANK
    assert scenarios.ACTIVE.primary_dataset(Benchmark.LLM) == DATASET_ID_LLM_BANK


def test_each_board_headline_reads_its_scenario_clean_dataset() -> None:
    assert scenarios.headline_dataset(Benchmark.S2S) == "s2s-workflow-bank-v1"
    assert scenarios.headline_dataset(Benchmark.LLM) == DATASET_ID_LLM_BANK
    assert scenarios.headline_dataset(Benchmark.TTS) is None


def test_the_s2s_headline_dataset_collects_latency() -> None:
    dataset = scenarios.headline_dataset(Benchmark.S2S)
    assert dataset is not None
    assert Metric.V2V in CONDITIONS[dataset].fetched


def test_only_text_runs_need_the_persona_setting() -> None:
    settings = Settings(coval_s2s_bank_test_set_id="T", coval_s2s_bank_instruction_metric_id="M")
    assert scenarios.ACTIVE.missing(settings, benchmark=Benchmark.S2S) == []
    assert scenarios.ACTIVE.missing(settings, benchmark=Benchmark.LLM) == [
        "coval_s2s_bank_persona_id"
    ]


def test_the_table_follows_the_slug_tuple_with_bank_first() -> None:
    assert tuple(s.slug for s in scenarios.SCENARIOS) == SCENARIO_SLUGS
    assert scenarios.SCENARIOS[0] is scenarios.BANK is scenarios.ACTIVE
    assert [s.label for s in scenarios.SCENARIOS] == [
        "Ultra Bank",
        "Happy Customer",
        "Happy Smile Clinic",
        "Ultra Bank · Instruction Adherence",
        "Ultra Bank · Workflow Adherence",
    ]


def test_only_bank_runs_over_text() -> None:
    happy_customer = scenarios.SCENARIOS[1]
    assert happy_customer.primary_dataset(Benchmark.S2S) == "s2s-happy-customer-v1"
    assert happy_customer.contract is None
    with pytest.raises(ValueError, match="does not run over text"):
        happy_customer.family(Benchmark.LLM)
    with pytest.raises(ValueError, match="no text-board settings"):
        happy_customer.ids(Settings())
