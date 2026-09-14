# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The shared scenario resolves the same ids for both modalities."""

from __future__ import annotations

from coval_bench import scenarios
from coval_bench.config import Settings
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.s2s.conditions import DATASET_ID_BANK, DATASET_ID_LLM_BANK


def test_bank_scenario_names_one_dataset_per_benchmark() -> None:
    assert scenarios.ACTIVE.primary_dataset(Benchmark.S2S) == DATASET_ID_BANK
    assert scenarios.ACTIVE.primary_dataset(Benchmark.LLM) == DATASET_ID_LLM_BANK


def test_only_text_runs_need_the_persona_setting() -> None:
    settings = Settings(coval_s2s_bank_test_set_id="T", coval_s2s_bank_instruction_metric_id="M")
    assert scenarios.ACTIVE.missing(settings, benchmark=Benchmark.S2S) == []
    assert scenarios.ACTIVE.missing(settings, benchmark=Benchmark.LLM) == [
        "coval_s2s_bank_persona_id"
    ]
