# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.speechmatics (SpeechmaticsProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.speechmatics import SpeechmaticsProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def make_provider(model: str = "default") -> SpeechmaticsProvider:
    return SpeechmaticsProvider(api_key=SecretStr("test-key-speechmatics"), model=model)


def _fake_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Happy path — default model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speechmatics_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("speechmatics", "events-success")
    provider = SpeechmaticsProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.speechmatics.ws_client.connect",
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
    assert result.complete_transcript is not None
    # Final transcripts: "hello world" + "how are you"
    assert "hello" in result.complete_transcript.lower()


# ---------------------------------------------------------------------------
# Happy path — enhanced model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speechmatics_enhanced(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("speechmatics", "events-success")
    provider = SpeechmaticsProvider(api_key=fake_api_key, model="enhanced")

    with patch(
        "coval_bench.providers.stt.speechmatics.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert result.complete_transcript is not None


# ---------------------------------------------------------------------------
# StartRecognition config
# ---------------------------------------------------------------------------


def test_start_recognition_config_default() -> None:
    p = make_provider("default")
    cfg = p._build_start_recognition_config(16000)
    assert cfg["message"] == "StartRecognition"
    assert cfg["audio_format"]["sample_rate"] == 16000
    assert "operating_point" not in cfg["transcription_config"]


def test_start_recognition_config_enhanced() -> None:
    p = make_provider("enhanced")
    cfg = p._build_start_recognition_config(16000)
    assert cfg["transcription_config"]["operating_point"] == "enhanced"


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


def test_provider_name_default() -> None:
    p = make_provider()
    assert p.name == "speechmatics"


def test_provider_name_enhanced() -> None:
    p = make_provider("enhanced")
    assert p.name == "speechmatics-enhanced"


# ---------------------------------------------------------------------------
# Invalid model
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Speechmatics model"):
        SpeechmaticsProvider(api_key=SecretStr("k"), model="bad-model")


# ---------------------------------------------------------------------------
# Failure path — empty stream (no RecognitionStarted → never proceeds)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speechmatics_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    # Provide RecognitionStarted so the handshake completes, then nothing
    events = [
        {"message": "RecognitionStarted", "id": "x"},
        {"message": "EndOfTranscript"},
    ]
    provider = SpeechmaticsProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.speechmatics.ws_client.connect",
        return_value=_fake_connect(events),
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


# ---------------------------------------------------------------------------
# Failure path — error event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speechmatics_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = [
        {"message": "RecognitionStarted", "id": "x"},
        {"message": "Error", "reason": "Authentication failed"},
    ]
    provider = SpeechmaticsProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.speechmatics.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "Authentication" in result.error
