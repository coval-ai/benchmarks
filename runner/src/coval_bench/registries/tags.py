# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Model tag vocabulary surfaced as leaderboard filters."""

from __future__ import annotations

from enum import StrEnum


class TagCategory(StrEnum):
    """Grouping for model tags, used to organize leaderboard filters."""

    CAPABILITY = "capability"


class ModelTag(StrEnum):
    """Filterable/sortable model attributes surfaced on the leaderboard."""

    REALTIME = "realtime"


TAG_CATEGORIES: dict[ModelTag, TagCategory] = {
    ModelTag.REALTIME: TagCategory.CAPABILITY,
}

if TAG_CATEGORIES.keys() != set(ModelTag):
    _missing = ", ".join(sorted(set(ModelTag) - TAG_CATEGORIES.keys()))
    raise RuntimeError(f"TAG_CATEGORIES is missing categories for: {_missing}")
