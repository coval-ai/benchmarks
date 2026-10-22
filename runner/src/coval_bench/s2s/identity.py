# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Which dataset a Coval run's rows belong to, from its test set and persona."""

from __future__ import annotations

import hashlib
import importlib.resources
from collections.abc import Mapping
from typing import TYPE_CHECKING

import structlog

from coval_bench import scenarios
from coval_bench.registries import Metric
from coval_bench.s2s.conditions import (
    DATASET_ID,
    FAMILY_MULTITURN,
    Condition,
    condition_for,
    dataset_id_for,
)

if TYPE_CHECKING:
    from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.s2s.identity")


def packaged_dataset_sha256() -> str:
    """SHA-256 of the packaged S2S manifest."""
    try:
        ref = importlib.resources.files("coval_bench.datasets.manifests").joinpath(
            f"{DATASET_ID}.json"
        )
        return hashlib.sha256(ref.read_bytes()).hexdigest()
    except Exception:
        logger.warning("dataset_sha256_failed", dataset_id=DATASET_ID, exc_info=True)
        return "unknown"


def fingerprint_for(provenance: str) -> str:
    """Return a stable 64-hex dataset fingerprint.

    Packaged datasets already carry a content SHA. Coval-hosted S2S datasets
    carry immutable test-set/persona provenance in the legacy run column, so
    fingerprint that identifier without changing the legacy representation.
    """
    if len(provenance) == 64 and all(character in "0123456789abcdef" for character in provenance):
        return provenance
    return hashlib.sha256(provenance.encode()).hexdigest()


class UnmappedPersona(ValueError):
    """A run's persona is absent from the condition map, so its rows have no home."""


def condition_for_persona(
    persona_id: str,
    persona_conditions: Mapping[str, Condition] | None,
    noisy_persona_id: str | None,
) -> Condition:
    """This run's caller condition.

    An unmapped persona raises rather than counting as clean: that pooling would be
    invisible in both the data and the logs. ``noisy_persona_id`` is the pre-map
    setting, honoured only while the map is unset so the two can deploy in either
    order.
    """
    if persona_conditions:
        try:
            return persona_conditions[persona_id]
        except KeyError:
            raise UnmappedPersona(f"persona {persona_id!r} has no condition mapped") from None
    if noisy_persona_id and persona_id == noisy_persona_id:
        return Condition.NOISY
    return Condition.CLEAN


def dataset_identity(
    test_set_id: str | None,
    persona_id: str = "",
    noisy_persona_id: str | None = None,
    *,
    family: str = FAMILY_MULTITURN,
    persona_conditions: Mapping[str, Condition] | None = None,
) -> tuple[str, str] | None:
    """Dataset id + provenance for one run's rows, or None when not ingested.

    Multi-turn runs are keyed to their Coval test set, not the single-turn SLURP
    manifest, so they get their own dataset id (never pooling with s2s-v1) and
    record the test-set id as provenance. Without a test set (legacy latency-only
    mode) the rows stay under the packaged s2s-v1 manifest.

    Non-clean conditions name the persona in their provenance so the rows stay
    self-describing.
    """
    if not test_set_id:
        return DATASET_ID, packaged_dataset_sha256()
    condition = condition_for_persona(persona_id, persona_conditions, noisy_persona_id)
    dataset_id = dataset_id_for(family, condition)
    if dataset_id is None:
        return None
    if condition is Condition.CLEAN:
        return dataset_id, test_set_id
    return dataset_id, f"{test_set_id}:{persona_id}"


def fetches_v2v(family: str) -> bool:
    """Whether *family*'s clean condition asks Coval for V2V latency."""
    dataset_id = dataset_id_for(family, Condition.CLEAN)
    return dataset_id is not None and Metric.V2V in condition_for(dataset_id).fetched


def parse_persona_conditions(raw: Mapping[str, str]) -> dict[str, Condition] | None:
    """Parse the persona -> condition setting, rejecting unknown condition names."""
    if not raw:
        return None
    parsed: dict[str, Condition] = {}
    for persona_id, name in raw.items():
        try:
            parsed[persona_id] = Condition(name)
        except ValueError:
            valid = ", ".join(c.value for c in Condition)
            raise RuntimeError(
                f"persona {persona_id!r} maps to condition {name!r}; expected one of {valid}"
            ) from None
    return parsed


def all_persona_conditions(settings: Settings) -> dict[str, str]:
    """The legacy persona map plus every scenario's callers, as condition names.

    Coval personas are workspace-level, so the same id may serve several
    scenarios; identical declarations merge. Two different conditions for one
    persona is a configuration error, not a tie.
    """
    merged = dict(settings.coval_s2s_condition_personas)
    owner = dict.fromkeys(merged, "coval_s2s_condition_personas")
    for slug, ids in settings.coval_s2s_scenarios.items():
        for condition, persona_id in ids.personas.items():
            previous = merged.get(persona_id)
            if previous is not None and previous != condition:
                raise RuntimeError(
                    f"persona {persona_id!r} is {previous!r} in {owner[persona_id]} but "
                    f"{condition!r} in coval_s2s_scenarios[{slug!r}]"
                )
            owner.setdefault(persona_id, f"coval_s2s_scenarios[{slug!r}]")
            merged[persona_id] = condition
    return merged


def all_persona_labels(settings: Settings) -> dict[str, str]:
    """Labels for the samples card: the legacy map plus the registry label of each
    scenario persona's condition."""
    registry = scenarios.load_personas().personas
    labels = dict(settings.s2s_persona_labels)
    for ids in settings.coval_s2s_scenarios.values():
        for condition, persona_id in ids.personas.items():
            persona = registry.get(condition)
            if persona is not None:
                labels.setdefault(persona_id, persona.label)
    return labels
