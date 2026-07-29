# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Settings, focused on placeholder-secret handling."""

from __future__ import annotations

from typing import Any

import structlog

from coval_bench.config import SECRET_PLACEHOLDER, Settings


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_placeholder_secretstr_treated_as_unset() -> None:
    settings = _settings(openai_api_key=SECRET_PLACEHOLDER)
    assert settings.openai_api_key is None


def test_placeholder_plain_str_treated_as_unset() -> None:
    settings = _settings(baseten_whisper_url=SECRET_PLACEHOLDER)
    assert settings.baseten_whisper_url is None


def test_placeholder_database_url_kept() -> None:
    # database_url is not nullable; the value survives so the connect failure
    # still names the placeholder.
    settings = _settings(database_url=SECRET_PLACEHOLDER)
    assert settings.database_url == SECRET_PLACEHOLDER


def test_placeholder_warns_per_field() -> None:
    with structlog.testing.capture_logs() as logs:
        _settings(openai_api_key=SECRET_PLACEHOLDER, database_url=SECRET_PLACEHOLDER)
    warned = {
        entry["setting"]
        for entry in logs
        if entry["event"] == "placeholder_secret" and entry["log_level"] == "warning"
    }
    assert warned == {"openai_api_key", "database_url"}


def test_real_values_pass_through_silently() -> None:
    with structlog.testing.capture_logs() as logs:
        settings = _settings(openai_api_key="sk-real", baseten_whisper_url="wss://example")
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-real"
    assert settings.baseten_whisper_url == "wss://example"
    assert not [entry for entry in logs if entry["event"] == "placeholder_secret"]
