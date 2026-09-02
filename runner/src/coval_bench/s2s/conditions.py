# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Which metrics each Coval caller condition carries.

A condition is one dataset id. ``required`` must be on the run or the run is a
fault; ``optional`` is written when present; anything named in neither is never
fetched, so its absence is silent rather than an alertable warning.

Dataset ids pair a family with a condition. The family is the Coval test set the
runs came from, so two sets' scenarios can never share a row key; the condition is
the caller persona's character within it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from coval_bench.registries import Benchmark, Metric

__all__ = [
    "DATASET_ID",
    "DATASET_ID_DENTAL",
    "DATASET_ID_DENTAL_ACCENTED",
    "DATASET_ID_DENTAL_NOISY",
    "DATASET_ID_HAPPYPATH",
    "DATASET_ID_HAPPYPATH_ACCENTED",
    "DATASET_ID_HAPPYPATH_NOISY",
    "DATASET_ID_LLM_DENTAL",
    "DATASET_ID_MULTITURN",
    "DATASET_ID_MULTITURN_NOISY",
    "DEFAULT_CONDITION",
    "FAMILY_DENTAL",
    "FAMILY_HAPPYPATH",
    "FAMILY_LLM_DENTAL",
    "FAMILY_MULTITURN",
    "Condition",
    "DatasetMetrics",
    "condition_for",
    "dataset_id_for",
]


class Condition(StrEnum):
    """One caller persona's character, independent of which test set ran it.

    ``SKIP`` marks a persona we know about and deliberately do not ingest, so the
    persona map can stay exhaustive — an *unmapped* persona faults instead of
    silently counting as clean.
    """

    CLEAN = "clean"
    NOISY = "noisy"
    ACCENTED = "accented"
    SKIP = "skip"


# The Coval test sets we ingest, as dataset-id prefixes. Their case counts differ
# by design: the shared set runs 15 cases across two clean voices and aggregates
# them, the happy-path set runs 30 across one, so the clean condition carries the
# same 30 conversations either way.
FAMILY_MULTITURN = "s2s-multiturn"  # customer-service, the shared board
FAMILY_HAPPYPATH = "s2s-happypath"
# Dental appointment booking, its own domain and its own 50 cases. Separate from
# happy-path on purpose: pooling two populations under one dataset id would break
# the one-dataset-id = one-condition = one-population anchor the metrics rest on.
FAMILY_DENTAL = "s2s-dental"
FAMILY_LLM_DENTAL = "llm-dental"

# Single-turn SLURP manifest (legacy, latency only) and the multi-turn Coval test
# set, split by caller condition so background noise never pools into the clean
# numbers.
DATASET_ID = "s2s-v1"
DATASET_ID_MULTITURN = "s2s-multiturn-v1"
DATASET_ID_MULTITURN_NOISY = "s2s-multiturn-noisy-v1"

DATASET_ID_HAPPYPATH = "s2s-happypath-v1"
DATASET_ID_HAPPYPATH_NOISY = "s2s-happypath-noisy-v1"
DATASET_ID_HAPPYPATH_ACCENTED = "s2s-happypath-accented-v1"

DATASET_ID_DENTAL = "s2s-dental-v1"
DATASET_ID_DENTAL_NOISY = "s2s-dental-noisy-v1"
DATASET_ID_DENTAL_ACCENTED = "s2s-dental-accented-v1"

# The same Coval dental test set driven over text: a separate population, never
# pooled with the voice rows.
DATASET_ID_LLM_DENTAL = "llm-dental-v1"

# Unlisted pairs are a configuration error, not a silent skip.
DATASET_IDS: dict[tuple[str, Condition], str | None] = {
    (FAMILY_MULTITURN, Condition.CLEAN): DATASET_ID_MULTITURN,
    (FAMILY_MULTITURN, Condition.NOISY): DATASET_ID_MULTITURN_NOISY,
    (FAMILY_HAPPYPATH, Condition.CLEAN): DATASET_ID_HAPPYPATH,
    (FAMILY_HAPPYPATH, Condition.NOISY): DATASET_ID_HAPPYPATH_NOISY,
    (FAMILY_HAPPYPATH, Condition.ACCENTED): DATASET_ID_HAPPYPATH_ACCENTED,
    (FAMILY_DENTAL, Condition.CLEAN): DATASET_ID_DENTAL,
    (FAMILY_DENTAL, Condition.NOISY): DATASET_ID_DENTAL_NOISY,
    (FAMILY_DENTAL, Condition.ACCENTED): DATASET_ID_DENTAL_ACCENTED,
    (FAMILY_LLM_DENTAL, Condition.CLEAN): DATASET_ID_LLM_DENTAL,
    (FAMILY_LLM_DENTAL, Condition.NOISY): None,
    (FAMILY_LLM_DENTAL, Condition.ACCENTED): None,
}


def dataset_id_for(family: str, condition: Condition) -> str | None:
    """The dataset id for *condition* within *family*, or None when not ingested."""
    if condition is Condition.SKIP:
        return None
    try:
        return DATASET_IDS[(family, condition)]
    except KeyError:
        raise ValueError(f"no dataset id for family {family!r} condition {condition!r}") from None


class DatasetMetrics(BaseModel, frozen=True):
    """One condition's metric contract.

    ``required`` is a single metric because it doubles as the population anchor:
    every optional metric's conversation ids must match it to be written.
    """

    benchmark: Benchmark = Benchmark.S2S
    required: Metric
    optional: frozenset[Metric] = frozenset()
    local: frozenset[Metric] = frozenset()

    @property
    def fetched(self) -> frozenset[Metric]:
        """Every metric worth asking Coval for; the rest are excluded."""
        return self.optional | {self.required}


# Background noise sits in the lane Silero VAD uses for turn boundaries, so V2V
# is omitted for the noisy caller: not measured, not written, not warned about.
CONDITIONS: dict[str, DatasetMetrics] = {
    DATASET_ID_MULTITURN: DatasetMetrics(
        required=Metric.V2V,
        optional=frozenset({Metric.INSTRUCTION_FOLLOWING, Metric.INTERRUPTION_RATE}),
    ),
    DATASET_ID_MULTITURN_NOISY: DatasetMetrics(
        required=Metric.INSTRUCTION_FOLLOWING,
        optional=frozenset({Metric.INTERRUPTION_RATE}),
    ),
    DATASET_ID_HAPPYPATH: DatasetMetrics(
        required=Metric.V2V,
        optional=frozenset({Metric.INSTRUCTION_FOLLOWING, Metric.INTERRUPTION_RATE}),
    ),
    DATASET_ID_HAPPYPATH_NOISY: DatasetMetrics(
        required=Metric.INSTRUCTION_FOLLOWING,
        optional=frozenset({Metric.INTERRUPTION_RATE}),
    ),
    # These runs were scored without V2V attached, so anchoring on it would skip
    # every one with ``required_metric_absent``. Move the anchor back if they are
    # ever re-scored with it.
    DATASET_ID_HAPPYPATH_ACCENTED: DatasetMetrics(
        required=Metric.INSTRUCTION_FOLLOWING,
        optional=frozenset({Metric.INTERRUPTION_RATE}),
    ),
    DATASET_ID_DENTAL: DatasetMetrics(
        required=Metric.V2V,
        optional=frozenset({Metric.INSTRUCTION_FOLLOWING, Metric.INTERRUPTION_RATE}),
    ),
    DATASET_ID_DENTAL_NOISY: DatasetMetrics(
        required=Metric.INSTRUCTION_FOLLOWING,
        optional=frozenset({Metric.INTERRUPTION_RATE}),
    ),
    # Anchored like its happy-path sibling: the same accented persona's runs are
    # scored without V2V there, and anchoring on a metric the runs lack would
    # skip every one of them and leave the condition silently empty.
    DATASET_ID_DENTAL_ACCENTED: DatasetMetrics(
        required=Metric.INSTRUCTION_FOLLOWING,
        optional=frozenset({Metric.INTERRUPTION_RATE}),
    ),
    DATASET_ID_LLM_DENTAL: DatasetMetrics(
        benchmark=Benchmark.LLM,
        required=Metric.INSTRUCTION_FOLLOWING,
        local=frozenset({Metric.TTFT}),
    ),
}

# Pre-scoping behaviour, for any dataset without an entry above.
DEFAULT_CONDITION = DatasetMetrics(
    required=Metric.V2V,
    optional=frozenset({Metric.INSTRUCTION_FOLLOWING}),
)


def condition_for(dataset_id: str) -> DatasetMetrics:
    """The metric contract for *dataset_id*."""
    return CONDITIONS.get(dataset_id, DEFAULT_CONDITION)
