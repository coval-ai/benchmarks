# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The scenario every LLM agent runs, named once.

A suite pairs the contract the proxy serves with the Coval test set, judge, and
persona the runs use, so the agent definition, the run template, and the fetch
all read the same ids.
"""

from __future__ import annotations

from dataclasses import dataclass

from coval_bench.config import Settings
from coval_bench.s2s.conditions import FAMILY_LLM_BANK


@dataclass(frozen=True)
class SuiteIds:
    test_set_id: str | None
    instruction_metric_id: str | None
    persona_id: str | None


@dataclass(frozen=True)
class Suite:
    contract: str
    family: str
    test_set_id_attr: str
    instruction_metric_id_attr: str
    persona_id_attr: str

    @property
    def id_attrs(self) -> tuple[str, str, str]:
        return (self.test_set_id_attr, self.instruction_metric_id_attr, self.persona_id_attr)

    def ids(self, settings: Settings) -> SuiteIds:
        values = [(getattr(settings, attr) or "").strip() or None for attr in self.id_attrs]
        return SuiteIds(*values)

    def missing(self, settings: Settings) -> list[str]:
        """The setting names still unset or blank, in declaration order."""
        ids = self.ids(settings)
        return [
            attr
            for attr, value in zip(
                self.id_attrs,
                (ids.test_set_id, ids.instruction_metric_id, ids.persona_id),
                strict=True,
            )
            if value is None
        ]


BANK = Suite(
    contract="bank",
    family=FAMILY_LLM_BANK,
    test_set_id_attr="coval_s2s_bank_test_set_id",
    instruction_metric_id_attr="coval_s2s_bank_instruction_metric_id",
    persona_id_attr="coval_s2s_bank_persona_id",
)

ACTIVE = BANK
