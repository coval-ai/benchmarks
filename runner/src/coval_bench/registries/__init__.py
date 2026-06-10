# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Code-as-source-of-truth registries of leaderboard display metadata.

Deliberately dependency-light: safe to import from the API, the db layer,
and the orchestrator without pulling in metric-computation dependencies.
"""

from coval_bench.registries.metrics import (
    METRIC_SPECS,
    Metric,
    MetricDirection,
    MetricSpec,
)
from coval_bench.registries.tags import TAG_CATEGORIES, ModelTag, TagCategory

__all__ = [
    "METRIC_SPECS",
    "Metric",
    "MetricDirection",
    "MetricSpec",
    "TAG_CATEGORIES",
    "ModelTag",
    "TagCategory",
]
