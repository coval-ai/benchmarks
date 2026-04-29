# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.assemblyai (AssemblyAIProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.assemblyai import AssemblyAIProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def make_provider() -> AssemblyAIProvider:
    return AssemblyAIProvider(api_key=SecretStr("test-key-assemblyai"))


def _fake_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemblyai_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("assemblyai", "events-success")
    provider = AssemblyAIProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.assemblyai.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
            audio_duration=3.0,
        )

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert "hello" in result.first_token_content.lower()
    assert result.complete_transcript is not None
    assert "hello world how are you" in result.complete_transcript


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    p = make_provider()
    assert p.name == "assemblyai-universal-streaming"


# ---------------------------------------------------------------------------
# Invalid model
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid AssemblyAI model"):
        AssemblyAIProvider(api_key=SecretStr("k"), model="bad-model")


# ---------------------------------------------------------------------------
# Invalid sample rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemblyai_wrong_sample_rate(audio_pcm_bytes: bytes) -> None:
    provider = make_provider()
    with pytest.raises(ValueError, match="16 kHz"):
        await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=8000,
        )


# ---------------------------------------------------------------------------
# Failure path — empty stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemblyai_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = AssemblyAIProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.assemblyai.ws_client.connect",
        return_value=_fake_connect([]),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.complete_transcript is None
    assert result.ttft_seconds is None
