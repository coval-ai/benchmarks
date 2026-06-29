# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Code-as-source-of-truth registries of leaderboard display metadata.

Deliberately dependency-light: safe to import from the API, the db layer,
and the orchestrator without pulling in metric-computation dependencies.
"""

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.metrics import (
    METRIC_SPECS,
    Metric,
    MetricDirection,
    MetricSpec,
)
from coval_bench.registries.models import (
    MODEL_REGISTRY,
    ModelStatus,
    RegisteredModel,
    Tenancy,
)
from coval_bench.registries.tags import (
    CATEGORY_LABELS,
    PROVIDER_VALUED_CATEGORIES,
    TAG_CATEGORIES,
    ModelTag,
    TagCategory,
    tag_value_label,
)

__all__ = [
    "Benchmark",
    "METRIC_SPECS",
    "MODEL_REGISTRY",
    "Metric",
    "MetricDirection",
    "MetricSpec",
    "ModelStatus",
    "RegisteredModel",
    "Tenancy",
    "CATEGORY_LABELS",
    "PROVIDER_VALUED_CATEGORIES",
    "TAG_CATEGORIES",
    "ModelTag",
    "TagCategory",
    "tag_value_label",
]
