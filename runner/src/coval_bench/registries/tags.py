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


# Display label per category.
CATEGORY_LABELS: dict[TagCategory, str] = {
    TagCategory.TYPE: "Type",
    TagCategory.MODE: "Mode",
    TagCategory.HOST: "Host",
    TagCategory.LAB: "Lab",
    TagCategory.FEATURES: "Features",
    TagCategory.SOURCE: "Source",
    TagCategory.TENANCY: "Tenancy",
}

if CATEGORY_LABELS.keys() != set(TagCategory):
    _missing = ", ".join(sorted(set(TagCategory) - CATEGORY_LABELS.keys()))
    raise RuntimeError(f"CATEGORY_LABELS is missing labels for: {_missing}")

# Categories whose values are provider/creator ids; the client formats them.
PROVIDER_VALUED_CATEGORIES: frozenset[TagCategory] = frozenset({TagCategory.HOST, TagCategory.LAB})

# Value labels that aren't a plain capitalization.
_VALUE_LABELS: dict[str, str] = {ModelTag.VAD.value: "VAD"}


def tag_value_label(category: TagCategory, value: str) -> str:
    """Display label for a tag value. Provider-valued categories keep the raw id."""
    if category in PROVIDER_VALUED_CATEGORIES:
        return value
    if category is TagCategory.TYPE:
        return value.upper()
    return _VALUE_LABELS.get(value, value.capitalize())
