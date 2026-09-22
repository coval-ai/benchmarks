# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The scenarios every benchmark runs, named once for both modalities."""

from __future__ import annotations

from dataclasses import dataclass

from coval_bench.config import Settings
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.s2s.conditions import (
    FAMILY_LLM_BANK,
    SCENARIO_SLUGS,
    Condition,
    dataset_id_for,
    scenario_family,
)


@dataclass(frozen=True)
class ScenarioIds:
    test_set_id: str | None
    instruction_metric_id: str | None
    persona_id: str | None


@dataclass(frozen=True)
class Scenario:
    """One instruction-following domain: its slug names the family and every dataset id.

    Voice agents find their Coval ids in ``Settings.coval_s2s_scenarios`` under the
    slug. The text board runs one scenario only, wired through the ``*_attr``
    settings names; they stay None on the others.
    """

    slug: str
    label: str
    # The prompt package under ``scenarios/`` the text board serves; None when the
    # scenario's prompt lives only in its Coval agents.
    contract: str | None = None
    llm_family: str | None = None
    test_set_id_attr: str | None = None
    instruction_metric_id_attr: str | None = None
    persona_id_attr: str | None = None

    def family(self, benchmark: Benchmark) -> str:
        if benchmark is Benchmark.S2S:
            return scenario_family(self.slug)
        if self.llm_family is None:
            raise ValueError(f"{self.slug} does not run over text")
        return self.llm_family

    def primary_dataset(self, benchmark: Benchmark) -> str:
        dataset_id = dataset_id_for(self.family(benchmark), Condition.CLEAN)
        if dataset_id is None:
            raise ValueError(f"{self.family(benchmark)} has no clean dataset id")
        return dataset_id

    @property
    def id_attrs(self) -> tuple[str, str, str]:
        attrs = (self.test_set_id_attr, self.instruction_metric_id_attr, self.persona_id_attr)
        if None in attrs:
            raise ValueError(f"{self.slug} has no text-board settings")
        return (str(attrs[0]), str(attrs[1]), str(attrs[2]))

    def ids(self, settings: Settings) -> ScenarioIds:
        values = [(getattr(settings, attr) or "").strip() or None for attr in self.id_attrs]
        return ScenarioIds(*values)

    def missing(self, settings: Settings, *, benchmark: Benchmark) -> list[str]:
        # Voice runs pick personas through the condition map, so only text needs one.
        ids = self.ids(settings)
        test_set_id_attr, instruction_metric_id_attr, persona_id_attr = self.id_attrs
        required = [
            (test_set_id_attr, ids.test_set_id),
            (instruction_metric_id_attr, ids.instruction_metric_id),
        ]
        if benchmark is Benchmark.LLM:
            required.append((persona_id_attr, ids.persona_id))
        return [attr for attr, value in required if value is None]


# Per-slug details; the slug tuple in ``s2s.conditions`` is the one list of scenarios.
_DETAILS: dict[str, dict[str, str]] = {
    "bank": {
        "label": "Ultra Bank",
        "contract": "bank",
        "llm_family": FAMILY_LLM_BANK,
        "test_set_id_attr": "coval_s2s_bank_test_set_id",
        "instruction_metric_id_attr": "coval_s2s_bank_instruction_metric_id",
        "persona_id_attr": "coval_s2s_bank_persona_id",
    },
    "happy-customer": {"label": "Happy Customer"},
    "happy-smile": {"label": "Happy Smile Clinic"},
}

# Every voice scenario, in board order; the first is the headline.
SCENARIOS: tuple[Scenario, ...] = tuple(
    Scenario(slug=slug, **_DETAILS[slug]) for slug in SCENARIO_SLUGS
)
BANK = SCENARIOS[0]

# The headline board and the text board.
ACTIVE = BANK
