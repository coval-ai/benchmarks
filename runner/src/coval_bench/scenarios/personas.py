# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The caller personas: presentation and dataset binding, declared once in ``personas.json``."""

from __future__ import annotations

import functools
import importlib.resources
import json
import re
from collections.abc import Mapping
from typing import Literal

from pydantic import Field, model_validator

from coval_bench.scenarios.annotated import AnnotatedModel

Anchor = Literal["latency", "judge"]

_SLUG = r"^[a-z][a-z0-9_]*$"
_DATASET_ID = r"^[a-z0-9][a-z0-9-]*$"


class Persona(AnnotatedModel):
    """How one caller persona is shown: label, description, and axis position."""

    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    order: int = Field(ge=0)


class PersonaBinding(AnnotatedModel):
    """Where one persona's runs land within a family, and which metric anchors them."""

    dataset_id: str = Field(pattern=_DATASET_ID)
    anchor: Anchor


class PersonaRegistry(AnnotatedModel):
    """Every declared persona plus, per family, the personas that run and their dataset ids."""

    personas: Mapping[str, Persona]
    bindings: Mapping[str, Mapping[str, PersonaBinding]]

    @model_validator(mode="after")
    def _consistent(self) -> PersonaRegistry:
        bad = sorted(slug for slug in self.personas if not re.match(_SLUG, slug))
        if bad:
            raise ValueError(f"persona slug(s) {bad} must match {_SLUG}")
        orders = sorted(p.order for p in self.personas.values())
        if orders != list(range(len(orders))):
            raise ValueError(
                f"persona orders must be 0..{len(orders) - 1} with no gaps, got {orders}"
            )
        bound_by: dict[str, str] = {}
        for family, slugs in self.bindings.items():
            for slug, binding in slugs.items():
                if slug not in self.personas:
                    raise ValueError(f"family {family!r} binds undeclared persona {slug!r}")
                if binding.dataset_id in bound_by:
                    raise ValueError(
                        f"dataset id {binding.dataset_id!r} is bound twice: "
                        f"{bound_by[binding.dataset_id]} and {family}/{slug}"
                    )
                bound_by[binding.dataset_id] = f"{family}/{slug}"
        return self


@functools.cache
def load_personas() -> PersonaRegistry:
    """Parse and validate ``personas.json``; cached for the process lifetime."""
    raw = importlib.resources.files("coval_bench.scenarios").joinpath("personas.json").read_bytes()
    return PersonaRegistry.model_validate(json.loads(raw))


def binding_for_dataset(dataset_id: str) -> tuple[str, Persona, PersonaBinding] | None:
    """The slug, persona and binding a dataset id is bound to, or None when no family binds it."""
    registry = load_personas()
    for slugs in registry.bindings.values():
        for slug, binding in slugs.items():
            if binding.dataset_id == dataset_id:
                return slug, registry.personas[slug], binding
    return None


def persona_for_dataset(dataset_id: str) -> tuple[str, Persona] | None:
    """The slug and persona a dataset id is bound to, or None when no family binds it."""
    bound = binding_for_dataset(dataset_id)
    return None if bound is None else bound[:2]
