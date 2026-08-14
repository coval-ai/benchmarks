# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Which metrics each S2S caller condition carries.

A condition is one dataset id. ``required`` must be on the run or the run is a
fault; ``optional`` is written when present; anything named in neither is never
fetched, so its absence is silent rather than an alertable warning.
"""

from __future__ import annotations

from pydantic import BaseModel

from coval_bench.registries import Metric

__all__ = [
    "DATASET_ID",
    "DATASET_ID_MULTITURN",
    "DATASET_ID_MULTITURN_NOISY",
    "DEFAULT_CONDITION",
    "DatasetMetrics",
    "condition_for",
]

# Single-turn SLURP manifest (legacy, latency only) and the multi-turn Coval test
# set, split by caller condition so background noise never pools into the clean
# numbers.
DATASET_ID = "s2s-v1"
DATASET_ID_MULTITURN = "s2s-multiturn-v1"
DATASET_ID_MULTITURN_NOISY = "s2s-multiturn-noisy-v1"


class DatasetMetrics(BaseModel, frozen=True):
    """One condition's metric contract.

    ``required`` is a single metric because it doubles as the population anchor:
    every optional metric's conversation ids must match it to be written.
    """

    required: Metric
    optional: frozenset[Metric] = frozenset()

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
}

# Pre-scoping behaviour, for any dataset without an entry above.
DEFAULT_CONDITION = DatasetMetrics(
    required=Metric.V2V,
    optional=frozenset({Metric.INSTRUCTION_FOLLOWING}),
)


def condition_for(dataset_id: str) -> DatasetMetrics:
    """The metric contract for *dataset_id*."""
    return CONDITIONS.get(dataset_id, DEFAULT_CONDITION)
