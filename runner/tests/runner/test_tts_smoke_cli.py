# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``coval-bench tts-smoke`` CLI subcommand.

The CLI is the workhorse for live one-shot provider verification. It must:
- emit a single line of well-formed JSON to stdout,
- exit 0 when the provider returns a valid WAV with no error,
- exit 1 when the provider returns ``error != None``,
- exit 2 when the requested provider name isn't registered.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from coval_bench.__main__ import cli
from coval_bench.providers.base import TTSResult


@pytest.fixture(autouse=True)
def _stub_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings() requires DATABASE_URL — stub one for the smoke CLI."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://runner:password@localhost:5432/benchmarks",
    )
    # Reset the lru_cache so each test gets a fresh Settings.
    from coval_bench.config import get_settings

    get_settings.cache_clear()


def _make_wav_path() -> Path:
    """Return a path to a small non-empty file (proxy for a WAV)."""
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.write(fd, b"RIFF" + b"\x00" * 100)
    os.close(fd)
    return Path(name)


def _ok_provider() -> MagicMock:
    """Mock TTS provider class whose .synthesize returns a valid TTSResult."""
    audio_path = _make_wav_path()
    instance = MagicMock()
    instance.synthesize = AsyncMock(
        return_value=TTSResult(
            provider="openai",
            model="tts-1-hd",
            voice="alloy",
            ttfa_ms=412.3,
            audio_path=audio_path,
            error=None,
        )
    )
    cls = MagicMock(return_value=instance)
    return cls


def _err_provider() -> MagicMock:
    """Mock TTS provider class whose .synthesize returns an error result."""
    instance = MagicMock()
    instance.synthesize = AsyncMock(
        return_value=TTSResult(
            provider="openai",
            model="tts-1-hd",
            voice="alloy",
            ttfa_ms=None,
            audio_path=None,
            error="connection refused",
        )
    )
    cls = MagicMock(return_value=instance)
    return cls


def test_tts_smoke_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path → exit 0, JSON line with ok=true and audio_bytes>0."""
    fake_registry = {"openai": _ok_provider()}
    monkeypatch.setattr("coval_bench.providers.tts.TTS_PROVIDERS", fake_registry)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tts-smoke",
            "--provider",
            "openai",
            "--model",
            "tts-1-hd",
            "--voice",
            "alloy",
            "--text",
            "Hello world.",
        ],
    )
    assert result.exit_code == 0, result.output
    line = result.output.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "tts_smoke"
    assert payload["provider"] == "openai"
    assert payload["model"] == "tts-1-hd"
    assert payload["voice"] == "alloy"
    assert payload["ttfa_ms"] == 412.3
    assert payload["error"] is None
    assert payload["ok"] is True
    assert payload["audio_bytes"] is not None
    assert payload["audio_bytes"] > 0


def test_tts_smoke_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider returns error → exit 1, JSON ok=false."""
    fake_registry = {"openai": _err_provider()}
    monkeypatch.setattr("coval_bench.providers.tts.TTS_PROVIDERS", fake_registry)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tts-smoke",
            "--provider",
            "openai",
            "--model",
            "tts-1-hd",
            "--voice",
            "alloy",
        ],
    )
    assert result.exit_code == 1, result.output
    line = result.output.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["ok"] is False
    assert payload["error"] == "connection refused"
    assert payload["audio_path"] is None


def test_tts_smoke_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown provider name → exit 2, no JSON line on stdout."""
    monkeypatch.setattr("coval_bench.providers.tts.TTS_PROVIDERS", {"openai": _ok_provider()})

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tts-smoke",
            "--provider",
            "definitely-not-real",
            "--model",
            "tts-1-hd",
            "--voice",
            "alloy",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "Unknown TTS provider" in result.output
