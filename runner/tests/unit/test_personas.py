# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The persona registry agrees with the condition tables and rejects drift."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coval_bench.registries import Metric
from coval_bench.s2s import conditions
from coval_bench.scenarios import PersonaRegistry, binding_for_dataset, load_personas

_ANCHOR_METRIC = {"latency": Metric.V2V, "judge": Metric.INSTRUCTION_FOLLOWING}


def test_bank_personas_are_declared_in_difficulty_order() -> None:
    registry = load_personas()
    by_order = sorted(registry.personas, key=lambda slug: registry.personas[slug].order)
    assert by_order == ["clean", "low", "medium", "hard", "extra_hard"]
    assert registry.personas["clean"].order == 0


def test_every_binding_matches_the_condition_tables() -> None:
    """The file and conditions.py describe the same dataset ids and anchors."""
    for family, slugs in load_personas().bindings.items():
        for slug, binding in slugs.items():
            condition = conditions.Condition(slug)
            assert conditions.dataset_id_for(family, condition) == binding.dataset_id
            contract = conditions.condition_for(binding.dataset_id)
            assert contract.required is _ANCHOR_METRIC[binding.anchor], f"{family}/{slug}"


def test_every_ingested_bank_dataset_is_bound() -> None:
    bound = {b.dataset_id for b in load_personas().bindings[conditions.FAMILY_BANK].values()}
    ingested = {
        dataset_id
        for (family, _condition), dataset_id in conditions.DATASET_IDS.items()
        if family == conditions.FAMILY_BANK and dataset_id is not None
    }
    assert bound == ingested


def test_binding_for_dataset_carries_the_anchor() -> None:
    bound = binding_for_dataset(conditions.DATASET_ID_BANK)
    assert bound is not None
    slug, _persona, binding = bound
    assert (slug, binding.anchor) == ("clean", "latency")
    hard = binding_for_dataset(conditions.DATASET_ID_BANK_HARD)
    assert hard is not None
    assert (hard[0], hard[1].label, hard[2].anchor) == ("hard", "Hard", "judge")
    assert binding_for_dataset(conditions.DATASET_ID_DENTAL) is None
    assert binding_for_dataset("no-such-dataset") is None


def _registry(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "personas": {
            "clean": {"label": "Clean", "description": "Baseline.", "order": 0},
            "hard": {"label": "Hard", "description": "Poor cell audio.", "order": 1},
        },
        "bindings": {"s2s-bank": {"clean": {"dataset_id": "s2s-bank-v1", "anchor": "latency"}}},
    }
    base.update(overrides)
    return base


def test_registry_accepts_rationale_keys_and_rejects_typos() -> None:
    ok = _registry()
    ok["personas"]["hard"]["_coval"] = "Hard Difficulty"  # type: ignore[index]
    PersonaRegistry.model_validate(ok)
    typo = _registry()
    typo["personas"]["hard"]["lable"] = "Hard"  # type: ignore[index]
    with pytest.raises(ValidationError, match="unknown key"):
        PersonaRegistry.model_validate(typo)


@pytest.mark.parametrize(
    ("bindings", "message"),
    [
        (
            {"s2s-bank": {"noisy": {"dataset_id": "s2s-bank-noisy-v1", "anchor": "judge"}}},
            "undeclared",
        ),
        (
            {
                "s2s-bank": {"clean": {"dataset_id": "s2s-bank-v1", "anchor": "latency"}},
                "llm-bank": {"hard": {"dataset_id": "s2s-bank-v1", "anchor": "judge"}},
            },
            "bound twice",
        ),
    ],
)
def test_registry_rejects_inconsistent_bindings(bindings: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PersonaRegistry.model_validate(_registry(bindings=bindings))


def test_registry_rejects_gapped_orders() -> None:
    gapped = _registry()
    gapped["personas"]["hard"]["order"] = 5  # type: ignore[index]
    with pytest.raises(ValidationError, match="no gaps"):
        PersonaRegistry.model_validate(gapped)
