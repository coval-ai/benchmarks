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
        )

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert "hello" in result.first_token_content.lower()
    assert result.complete_transcript == "hello world how are you"


# ---------------------------------------------------------------------------
# URL sanity
# ---------------------------------------------------------------------------


def test_websocket_url_contains_required_params() -> None:
    from coval_bench.providers.stt.assemblyai import _SPEECH_MODEL_MAP, _WS_BASE

    assert "streaming.assemblyai.com/v3/ws" in _WS_BASE
    # Every mapped speech_model value must be a non-empty API string
    for user_name, api_name in _SPEECH_MODEL_MAP.items():
        assert api_name, f"model {user_name!r} maps to an empty speech_model"
        assert "format_turns" not in api_name


@pytest.mark.asyncio
async def test_force_endpoint_sent_before_terminate(
    fake_api_key: SecretStr, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("coval_bench.providers.stt.assemblyai._FINAL_WAIT_S", 0.05)
    sent: list[Any] = []
    final = {"type": "Turn", "end_of_turn": True, "transcript": "hello world"}
    ws = FakeWebSocket([final], on_send=sent.append)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    provider = AssemblyAIProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.assemblyai.ws_client.connect", return_value=cm
    ) as mock_connect:
        result = await provider.measure_ttft(
            audio_data=b"\x00" * 640,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.01,
        )

    url = mock_connect.call_args.args[0]
    assert "end_of_turn_confidence_threshold=1.0" in url

    text = [m for m in sent if isinstance(m, str)]
    assert '"ForceEndpoint"' in text[-2]
    assert '"Terminate"' in text[-1]
    # The forced final is captured before the close (gate + response-capture path).
    assert result.audio_to_final_seconds is not None


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
    result = await provider.measure_ttft(
        audio_data=audio_pcm_bytes,
        channels=1,
        sample_width=2,
        sample_rate=8000,
    )

    assert result.error is not None
    assert "16 kHz" in result.error
    assert result.ttft_seconds is None


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


# ---------------------------------------------------------------------------
# audio_to_final_seconds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemblyai_audio_to_final_populated(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """audio_to_final_seconds is set when at least one end_of_turn event is received."""
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
        )

    assert result.audio_to_final_seconds is not None
    assert result.audio_to_final_seconds >= 0


@pytest.mark.asyncio
async def test_assemblyai_audio_to_final_none_when_no_end_of_turn(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """audio_to_final_seconds is None when no end_of_turn event arrives."""
    events: list[Any] = [
        {"type": "Begin", "id": "test-session", "expires_at": 9999999999},
        {
            "type": "Turn",
            "transcript": "hello world",
            "end_of_turn": False,
            "words": [{"text": "hello"}, {"text": "world"}],
        },
    ]
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
        )

    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_assemblyai_audio_to_final_uses_last_end_of_turn(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """audio_to_final_seconds reflects the LAST end_of_turn event (not the first).

    Verified by using a multi-turn fixture with two end_of_turn events and
    confirming that: (a) audio_to_final_seconds is populated, and (b) the
    complete_transcript contains segments from both turns, proving the receive
    loop did not exit after the first end_of_turn.
    """
    events = load_fixture_events("assemblyai", "events-multi-turn")
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
        )

    assert result.audio_to_final_seconds is not None
    assert result.audio_to_final_seconds >= 0
    # Both end_of_turn turns must appear — proves we processed all final events
    assert result.complete_transcript is not None
    assert "hello world" in result.complete_transcript
    assert "how are you" in result.complete_transcript


@pytest.mark.asyncio
async def test_assemblyai_multi_turn_complete_transcript(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """All end_of_turn transcripts are joined into complete_transcript (not just first)."""
    events = load_fixture_events("assemblyai", "events-multi-turn")
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
        )

    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
