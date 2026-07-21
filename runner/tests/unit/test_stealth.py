# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for coval_bench.registries.stealth.

The invariant under test: real upstream identities exist only in the parsed
``STEALTH_MODELS`` value — every derived registry entry carries aliases and
positional voice labels, and malformed input degrades to "no stealth models",
never to an error or a leak.
"""

from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.registries import (
    STEALTH_PROVIDER,
    Benchmark,
    ModelStatus,
    stealth_entries,
    stealth_upstreams,
)


def _settings(stealth_json: str | None) -> Settings:
    return Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        stealth_models=None if stealth_json is None else SecretStr(stealth_json),
    )


_VALID = json.dumps(
    {
        "stealth-01": {
            "benchmark": "TTS",
            "provider": "elevenlabs",
            "model": "real-model-id",
            "voice": "real-voice-pinned",
            "voices": ["real-voice-f", "real-voice-m"],
            "api_key": "sk-under-nda",
        },
        "stealth-02": {
            "benchmark": "STT",
            "provider": "deepgram",
            "model": "real-stt-model",
        },
    }
)


def test_unset_yields_nothing() -> None:
    settings = _settings(None)
    assert stealth_upstreams(settings) == {}
    assert stealth_entries(settings) == []


def test_valid_json_parses_upstreams() -> None:
    upstreams = stealth_upstreams(_settings(_VALID))
    assert set(upstreams) == {"stealth-01", "stealth-02"}
    tts = upstreams["stealth-01"]
    assert (tts.provider, tts.model) == ("elevenlabs", "real-model-id")
    assert tts.api_key is not None
    assert tts.api_key.get_secret_value() == "sk-under-nda"
    stt = upstreams["stealth-02"]
    assert (stt.benchmark, stt.api_key) == (Benchmark.STT, None)


def test_entries_carry_only_aliases() -> None:
    entries = {e.model: e for e in stealth_entries(_settings(_VALID))}
    tts = entries["stealth-01"]
    assert tts.provider == STEALTH_PROVIDER
    assert tts.status is ModelStatus.EARLY_ACCESS
    assert tts.arena_enabled is False
    assert tts.voice == "voice-0"
    assert tts.voices == ("voice-1", "voice-2")

    stt = entries["stealth-02"]
    assert (stt.voice, stt.voices) == (None, ())

    for entry in entries.values():
        dumped = entry.model_dump_json()
        assert "real-" not in dumped
        assert "elevenlabs" not in dumped
        assert "deepgram" not in dumped
        assert "sk-under-nda" not in dumped


@pytest.mark.parametrize(
    "raw",
    [
        "PLACEHOLDER_REPLACE_VIA_GCLOUD",
        "{not json",
        '["a", "list"]',
        '{"alias": {"provider": "x"}}',  # missing required fields
        '{"alias": {"benchmark": "S2S", "provider": "x", "model": "y"}}',  # S2S is fetch-only
        '{"alias": {"benchmark": "STT", "provider": "x", "model": "y", "extra": 1}}',
    ],
)
def test_malformed_input_yields_nothing(raw: str) -> None:
    settings = _settings(raw)
    assert stealth_upstreams(settings) == {}
    assert stealth_entries(settings) == []


def test_resolve_voice_maps_labels_to_real_ids() -> None:
    upstream = stealth_upstreams(_settings(_VALID))["stealth-01"]
    assert upstream.resolve_voice(None) is None
    assert upstream.resolve_voice("voice-0") == "real-voice-pinned"
    assert upstream.resolve_voice("voice-1") == "real-voice-f"
    assert upstream.resolve_voice("voice-2") == "real-voice-m"
