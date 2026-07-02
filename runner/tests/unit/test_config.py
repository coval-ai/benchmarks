# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Settings loading."""

from __future__ import annotations

import pytest

from coval_bench.config import Settings


def test_provider_keys_strip_surrounding_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INWORLD_API_KEY", "abc123\n")
    monkeypatch.setenv("OPENAI_API_KEY", " sk-test ")

    settings = Settings()

    assert settings.inworld_api_key is not None
    assert settings.inworld_api_key.get_secret_value() == "abc123"
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test"
