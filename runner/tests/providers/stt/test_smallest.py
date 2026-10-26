# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.smallest (SmallestSTTProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.stt.smallest import SmallestSTTProvider
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings


def make_provider() -> SmallestSTTProvider:
    return SmallestSTTProvider(make_settings())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smallest_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("smallest", "events-success")
    sent: list[Any] = []
    provider = SmallestSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events, sent),
    ) as connect:
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert "hello" in result.first_token_content.lower()
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5

    assert connect.call_args.args[0].endswith("?language=en&encoding=linear16&sample_rate=16000")
    assert connect.call_args.kwargs["additional_headers"] == {
        "Authorization": "Bearer test-api-key-00000000000000000000"
    }
    assert json.loads(sent[-1]) == {"type": "close_stream"}


# ---------------------------------------------------------------------------
# Provider name and model
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    p = make_provider()
    assert p.name == "smallest"


def test_provider_model() -> None:
    p = make_provider()
    assert p.model == "pulse"


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="smallest_api_key is required"):
        SmallestSTTProvider(make_settings(smallest_api_key=None))


# ---------------------------------------------------------------------------
# Empty stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smallest_empty_stream(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = SmallestSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect([]),
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
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    events = load_fixture_events("smallest", "events-success")
    provider = SmallestSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
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
    stt_settings: Settings, audio_pcm_bytes: bytes
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
    provider = SmallestSTTProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect",
        return_value=fake_connect(events),
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
    # The longest partial is the transcript when no final lands.
    assert result.complete_transcript == "hello"
