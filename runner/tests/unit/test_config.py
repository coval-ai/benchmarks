# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Settings placeholder-secret handling."""

from __future__ import annotations

from typing import Any

import structlog

from coval_bench.config import SECRET_PLACEHOLDER, Settings


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_placeholder_secrets_nulled_and_warned() -> None:
    with structlog.testing.capture_logs() as logs:
        settings = _settings(
            openai_api_key=SECRET_PLACEHOLDER,
            baseten_whisper_url=SECRET_PLACEHOLDER,
            database_url=SECRET_PLACEHOLDER,
        )
    assert settings.openai_api_key is None
    assert settings.baseten_whisper_url is None
    assert settings.database_url == SECRET_PLACEHOLDER  # not nullable: kept, warned
    warned = {entry["setting"] for entry in logs if entry["event"] == "placeholder_secret"}
    assert warned == {"openai_api_key", "baseten_whisper_url", "database_url"}


def test_real_values_pass_through_silently() -> None:
    with structlog.testing.capture_logs() as logs:
        settings = _settings(openai_api_key="sk-real", baseten_whisper_url="wss://example")
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-real"
    assert settings.baseten_whisper_url == "wss://example"
    assert not [entry for entry in logs if entry["event"] == "placeholder_secret"]
