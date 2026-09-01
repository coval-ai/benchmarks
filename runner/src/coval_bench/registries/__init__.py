# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Code-as-source-of-truth registries of leaderboard display metadata.

Deliberately dependency-light: safe to import from the API, the db layer,
and the orchestrator without pulling in metric-computation dependencies.
"""

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.metrics import (
    METRIC_EXCLUSIONS,
    METRIC_SPECS,
    METRIC_VALUE_CONTRACTS,
    SERIES_EXCLUDED_METRICS,
    Metric,
    MetricDirection,
    MetricSpec,
    MetricValueContract,
    MetricValueDefinition,
    MetricValueRole,
    is_metric_excluded,
    validate_metric_contract,
    validate_metric_values,
)
from coval_bench.registries.models import (
    MODEL_REGISTRY,
    Gender,
    Licensing,
    RegisteredModel,
    Source,
    Voice,
)
from coval_bench.registries.preprocessing import (
    SUPPORTED_PREPROCESSING_ARTIFACT_CONTRACTS,
    validate_preprocessing_artifact_contract,
)
from coval_bench.registries.tags import (
    CATEGORY_LABELS,
    PROVIDER_VALUED_CATEGORIES,
    TagCategory,
    tag_value_label,
)

__all__ = [
    "Benchmark",
    "METRIC_EXCLUSIONS",
    "METRIC_SPECS",
    "METRIC_VALUE_CONTRACTS",
    "SUPPORTED_PREPROCESSING_ARTIFACT_CONTRACTS",
    "SERIES_EXCLUDED_METRICS",
    "MODEL_REGISTRY",
    "Metric",
    "MetricDirection",
    "MetricSpec",
    "Gender",
    "MetricValueContract",
    "MetricValueDefinition",
    "MetricValueRole",
    "Licensing",
    "RegisteredModel",
    "Source",
    "Voice",
    "CATEGORY_LABELS",
    "PROVIDER_VALUED_CATEGORIES",
    "TagCategory",
    "is_metric_excluded",
    "validate_metric_contract",
    "validate_metric_values",
    "validate_preprocessing_artifact_contract",
    "tag_value_label",
]
