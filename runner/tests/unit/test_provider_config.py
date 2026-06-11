# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the provider matrix tag scaffolding."""

from __future__ import annotations

from coval_bench.registries import TAG_CATEGORIES, ModelTag
from coval_bench.runner.config import (
    DEFAULT_STT_MATRIX,
    DEFAULT_TTS_MATRIX,
    ProviderEntry,
)


def test_every_tag_has_a_category() -> None:
    assert TAG_CATEGORIES.keys() == set(ModelTag)


def test_entry_tag_defaults() -> None:
    entry = ProviderEntry(provider="deepgram", model="nova-3", enabled=True)
    assert entry.creator is None
    assert entry.tags == []


def test_matrix_entries_untagged_for_now() -> None:
    # Scaffolding only — no matrix entry carries tags or a creator override yet.
    for entry in DEFAULT_STT_MATRIX + DEFAULT_TTS_MATRIX:
        assert entry.creator is None
        assert entry.tags == []
