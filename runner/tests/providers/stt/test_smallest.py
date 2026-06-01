# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.smallest (SmallestSTTProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.smallest import SmallestSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def make_provider() -> SmallestSTTProvider:
    return SmallestSTTProvider(api_key=SecretStr("test-key-smallest"))


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
async def test_smallest_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("smallest", "events-success")
    provider = SmallestSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.smallest.ws_client.connect",
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
# Provider name and model
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    p = make_provider()
    assert p.name == "smallest"


def test_provider_model() -> None:
    p = make_provider()
    assert p.model == "pulse"


# ---------------------------------------------------------------------------
# Invalid model
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Smallest STT model"):
        SmallestSTTProvider(api_key=SecretStr("k"), model="bad-model")


# ---------------------------------------------------------------------------
# Empty stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smallest_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = SmallestSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.smallest.ws_client.connect",
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


# ---------------------------------------------------------------------------
# audio_to_final_seconds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_to_final_set_on_is_last(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events = load_fixture_events("smallest", "events-success")
    provider = SmallestSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.smallest.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.audio_to_final_seconds is not None
    assert result.audio_to_final_seconds >= 0


@pytest.mark.asyncio
async def test_audio_to_final_none_on_partial_only(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """audio_to_final_seconds is None when only partial (is_final=false) messages arrive."""
    events: list[Any] = [
        {
            "type": "transcription",
            "status": "success",
            "session_id": "sess_x",
            "transcript": "hello",
            "is_final": False,
            "is_last": False,
            "language": "en",
        }
    ]
    provider = SmallestSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.smallest.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.audio_to_final_seconds is None
    # TTFT is still captured from the partial message
    assert result.ttft_seconds is not None
