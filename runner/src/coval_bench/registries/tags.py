# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Model tag vocabulary surfaced as leaderboard filters."""

from __future__ import annotations

from enum import StrEnum


class TagCategory(StrEnum):
    """Faceted leaderboard filters. Within a facet tags OR; across facets they AND.

    TYPE/HOST/LAB are derived from registry columns and SOURCE/TENANCY from model
    attributes, all at the API boundary; only MODE and FEATURES draw their values
    from ``ModelTag``.
    """

    TYPE = "type"
    MODE = "mode"
    HOST = "host"
    LAB = "lab"
    FEATURES = "features"
    SOURCE = "source"
    TENANCY = "tenancy"


class ModelTag(StrEnum):
    """Curated model attributes surfaced as MODE and FEATURES filter chips."""

    REALTIME = "realtime"
    BATCH = "batch"
    MULTILINGUAL = "multilingual"
    VAD = "vad"


TAG_CATEGORIES: dict[ModelTag, TagCategory] = {
    ModelTag.REALTIME: TagCategory.MODE,
    ModelTag.BATCH: TagCategory.MODE,
    ModelTag.MULTILINGUAL: TagCategory.FEATURES,
    ModelTag.VAD: TagCategory.FEATURES,
}

if TAG_CATEGORIES.keys() != set(ModelTag):
    _missing = ", ".join(sorted(set(ModelTag) - TAG_CATEGORIES.keys()))
    raise RuntimeError(f"TAG_CATEGORIES is missing categories for: {_missing}")
