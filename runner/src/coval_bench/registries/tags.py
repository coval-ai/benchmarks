# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Leaderboard facets derived from model columns; FEATURES lives in the tags table."""

from __future__ import annotations

from enum import StrEnum


class TagCategory(StrEnum):
    """Faceted leaderboard filters. Within a facet tags OR; across facets they AND."""

    TYPE = "type"
    HOST = "host"
    CREATOR = "creator"
    FEATURES = "features"
    SOURCE = "source"
    LICENSING = "licensing"
    DEPLOYMENT = "deployment"
    REGION = "region"


# Display label per category.
CATEGORY_LABELS: dict[TagCategory, str] = {
    TagCategory.TYPE: "Type",
    TagCategory.HOST: "Host",
    TagCategory.CREATOR: "Creator",
    TagCategory.FEATURES: "Features",
    TagCategory.SOURCE: "Source",
    TagCategory.LICENSING: "Licensing",
    TagCategory.DEPLOYMENT: "Deployment",
    TagCategory.REGION: "Server location",
}

if CATEGORY_LABELS.keys() != set(TagCategory):
    _missing = ", ".join(sorted(set(TagCategory) - CATEGORY_LABELS.keys()))
    raise RuntimeError(f"CATEGORY_LABELS is missing labels for: {_missing}")

# Categories whose values are provider/creator ids; the client formats them.
PROVIDER_VALUED_CATEGORIES: frozenset[TagCategory] = frozenset(
    {TagCategory.HOST, TagCategory.CREATOR}
)

_REGION_LABELS: dict[str, str] = {
    "us": "US",
    "eu": "Europe",
    "asia": "Asia",
}

# Value labels that aren't a plain capitalization.
_VALUE_LABELS: dict[str, str] = {
    "shared-inference": "Shared inference",
    "dedicated-inference": "Dedicated inference",
    "official-api": "Official API",
}


def tag_value_label(category: TagCategory, value: str) -> str:
    """Display label for a derived facet value. Provider-valued categories keep the raw id."""
    if category in PROVIDER_VALUED_CATEGORIES:
        return value
    if category is TagCategory.TYPE:
        return value.upper()
    if category is TagCategory.REGION:
        return _REGION_LABELS.get(value, value)
    return _VALUE_LABELS.get(value, value.capitalize())
