# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Model tag vocabulary surfaced as leaderboard filters."""

from __future__ import annotations

from enum import StrEnum


class TagCategory(StrEnum):
    """Faceted leaderboard filters. Within a facet tags OR; across facets they AND.

    TYPE/HOST/CREATOR are derived from registry columns and SOURCE/LICENSING/
    DEPLOYMENT from model attributes, all at the API boundary; only MODE and
    FEATURES draw their values from ``ModelTag``.
    """

    TYPE = "type"
    MODE = "mode"
    HOST = "host"
    CREATOR = "creator"
    FEATURES = "features"
    SOURCE = "source"
    LICENSING = "licensing"
    DEPLOYMENT = "deployment"


class ModelTag(StrEnum):
    """Curated model attributes surfaced as MODE and FEATURES filter chips."""

    STREAMING = "streaming"
    BATCH = "batch"
    MULTILINGUAL = "multilingual"
    VAD = "vad"
    DIARIZATION = "diarization"
    TRANSLATION = "translation"
    CODE_SWITCHING = "code-switching"
    KEYTERM_BIASING = "keyterm-biasing"
    VOICE_CLONING = "voice-cloning"
    EMOTION_CONTROL = "emotion-control"
    STREAMING_INPUT = "streaming-input"
    CONVERSATIONAL_CONTEXT = "conversational-context"


TAG_CATEGORIES: dict[ModelTag, TagCategory] = {
    ModelTag.STREAMING: TagCategory.MODE,
    ModelTag.BATCH: TagCategory.MODE,
    ModelTag.MULTILINGUAL: TagCategory.FEATURES,
    ModelTag.VAD: TagCategory.FEATURES,
    ModelTag.DIARIZATION: TagCategory.FEATURES,
    ModelTag.TRANSLATION: TagCategory.FEATURES,
    ModelTag.CODE_SWITCHING: TagCategory.FEATURES,
    ModelTag.KEYTERM_BIASING: TagCategory.FEATURES,
    ModelTag.VOICE_CLONING: TagCategory.FEATURES,
    ModelTag.EMOTION_CONTROL: TagCategory.FEATURES,
    ModelTag.STREAMING_INPUT: TagCategory.FEATURES,
    ModelTag.CONVERSATIONAL_CONTEXT: TagCategory.FEATURES,
}

if TAG_CATEGORIES.keys() != set(ModelTag):
    _missing = ", ".join(sorted(set(ModelTag) - TAG_CATEGORIES.keys()))
    raise RuntimeError(f"TAG_CATEGORIES is missing categories for: {_missing}")


# Display label per category.
CATEGORY_LABELS: dict[TagCategory, str] = {
    TagCategory.TYPE: "Type",
    TagCategory.MODE: "Mode",
    TagCategory.HOST: "Host",
    TagCategory.CREATOR: "Creator",
    TagCategory.FEATURES: "Features",
    TagCategory.SOURCE: "Source",
    TagCategory.LICENSING: "Licensing",
    TagCategory.DEPLOYMENT: "Deployment",
}

if CATEGORY_LABELS.keys() != set(TagCategory):
    _missing = ", ".join(sorted(set(TagCategory) - CATEGORY_LABELS.keys()))
    raise RuntimeError(f"CATEGORY_LABELS is missing labels for: {_missing}")

# Categories whose values are provider/creator ids; the client formats them.
PROVIDER_VALUED_CATEGORIES: frozenset[TagCategory] = frozenset(
    {TagCategory.HOST, TagCategory.CREATOR}
)

# Value labels that aren't a plain capitalization.
_VALUE_LABELS: dict[str, str] = {
    "shared-inference": "Shared inference",
    "dedicated-inference": "Dedicated inference",
    "official-api": "Official API",
    ModelTag.VAD.value: "VAD",
    ModelTag.CODE_SWITCHING.value: "Code switching",
    ModelTag.KEYTERM_BIASING.value: "Keyterm biasing",
    ModelTag.VOICE_CLONING.value: "Voice cloning",
    ModelTag.EMOTION_CONTROL.value: "Emotion control",
    ModelTag.STREAMING_INPUT.value: "Streaming input",
    ModelTag.CONVERSATIONAL_CONTEXT.value: "Conversational context",
}


def tag_value_label(category: TagCategory, value: str) -> str:
    """Display label for a tag value. Provider-valued categories keep the raw id."""
    if category in PROVIDER_VALUED_CATEGORIES:
        return value
    if category is TagCategory.TYPE:
        return value.upper()
    return _VALUE_LABELS.get(value, value.capitalize())
