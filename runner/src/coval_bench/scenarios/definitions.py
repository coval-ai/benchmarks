# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The scenario every benchmark runs, named once for both modalities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from coval_bench.config import Settings
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.s2s.conditions import FAMILY_BANK, FAMILY_LLM_BANK, Condition, dataset_id_for


@dataclass(frozen=True)
class ScenarioIds:
    test_set_id: str | None
    instruction_metric_id: str | None
    persona_id: str | None


@dataclass(frozen=True)
class Scenario:
    contract: str
    families: Mapping[Benchmark, str]
    test_set_id_attr: str
    instruction_metric_id_attr: str
    persona_id_attr: str

    def family(self, benchmark: Benchmark) -> str:
        return self.families[benchmark]

    def primary_dataset(self, benchmark: Benchmark) -> str:
        dataset_id = dataset_id_for(self.family(benchmark), Condition.CLEAN)
        if dataset_id is None:
            raise ValueError(f"{self.family(benchmark)} has no clean dataset id")
        return dataset_id

    @property
    def id_attrs(self) -> tuple[str, str, str]:
        return (self.test_set_id_attr, self.instruction_metric_id_attr, self.persona_id_attr)

    def ids(self, settings: Settings) -> ScenarioIds:
        values = [(getattr(settings, attr) or "").strip() or None for attr in self.id_attrs]
        return ScenarioIds(*values)

    def missing(self, settings: Settings, *, benchmark: Benchmark) -> list[str]:
        # Voice runs pick personas through the condition map, so only text needs one.
        ids = self.ids(settings)
        required = [
            (self.test_set_id_attr, ids.test_set_id),
            (self.instruction_metric_id_attr, ids.instruction_metric_id),
        ]
        if benchmark is Benchmark.LLM:
            required.append((self.persona_id_attr, ids.persona_id))
        return [attr for attr, value in required if value is None]


BANK = Scenario(
    contract="bank",
    families={Benchmark.S2S: FAMILY_BANK, Benchmark.LLM: FAMILY_LLM_BANK},
    test_set_id_attr="coval_s2s_bank_test_set_id",
    instruction_metric_id_attr="coval_s2s_bank_instruction_metric_id",
    persona_id_attr="coval_s2s_bank_persona_id",
)

ACTIVE = BANK
