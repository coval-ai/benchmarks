# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Code-as-source-of-truth registries of leaderboard display metadata.

Deliberately dependency-light: safe to import from the API, the db layer,
and the orchestrator without pulling in metric-computation dependencies.
"""

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.metrics import (
    INTERNAL_METRICS,
    METRIC_EXCLUSIONS,
    METRIC_SPECS,
    SERIES_EXCLUDED_METRICS,
    Metric,
    MetricDirection,
    MetricSpec,
    is_metric_excluded,
)
from coval_bench.registries.models import (
    MODEL_REGISTRY,
    Licensing,
    ModelStatus,
    RegisteredModel,
    Source,
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
    "INTERNAL_METRICS",
    "METRIC_EXCLUSIONS",
    "METRIC_SPECS",
    "SERIES_EXCLUDED_METRICS",
    "MODEL_REGISTRY",
    "Metric",
    "MetricDirection",
    "MetricSpec",
    "Licensing",
    "ModelStatus",
    "RegisteredModel",
    "Source",
    "CATEGORY_LABELS",
    "PROVIDER_VALUED_CATEGORIES",
    "TAG_CATEGORIES",
    "ModelTag",
    "TagCategory",
    "is_metric_excluded",
    "tag_value_label",
]
