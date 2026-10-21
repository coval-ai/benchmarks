# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.assemblyai (AssemblyAIProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.stt.assemblyai import AssemblyAIProvider
from tests.providers.stt.conftest import fake_connect, load_fixture_events, make_settings

# A synthetic Pro id: the tests pin the id-to-wire rule, not any registry entry.
PRO_MODEL = "universal-1.2-pro"


def make_provider() -> AssemblyAIProvider:
    return AssemblyAIProvider(make_settings())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemblyai_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("assemblyai", "events-success")
    provider = AssemblyAIProvider(stt_settings)

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
async def test_universal_streaming_terminates_without_force_endpoint(
    stt_settings: Settings,
) -> None:
    """TTFS-excluded models end the session per the vendor's streaming-WER guide."""
    sent: list[Any] = []
    final = {"type": "Turn", "end_of_turn": True, "transcript": "hello world"}
    cm = fake_connect([final], sent)
    provider = AssemblyAIProvider(stt_settings)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect", return_value=cm
    ) as mock_connect:
        result = await provider.measure_ttft(
            audio_data=b"\x00" * 640,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.01,
        )

    url = mock_connect.call_args.args[0]
    assert "end_of_turn_confidence_threshold" not in url

    text = [m for m in sent if isinstance(m, str)]
    assert text == ['{"type": "Terminate"}']
    assert result.complete_transcript == "hello world"


@pytest.mark.asyncio
async def test_pro_force_endpoint_before_terminate(stt_settings: Settings) -> None:
    sent: list[Any] = []
    final = {"type": "Turn", "end_of_turn": True, "transcript": "hello world"}
    cm = fake_connect([final], sent)
    provider = AssemblyAIProvider(stt_settings, model=PRO_MODEL)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect", return_value=cm
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
    assert result.audio_to_final_seconds is not None


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    p = make_provider()
    assert p.name == "assemblyai-universal-streaming"


def test_provider_name_universal_streaming_multilingual() -> None:
    p = AssemblyAIProvider(make_settings(), model="universal-streaming-multilingual")
    assert p.name == "assemblyai-universal-streaming-multilingual"


@pytest.mark.asyncio
async def test_pro_url_uses_api_speech_model_and_voice_agent_config(
    stt_settings: Settings,
) -> None:
    """Any Pro id maps to the API speech_model and gets the min_latency voice-agent preset."""
    cm = fake_connect([{"type": "Turn", "end_of_turn": True, "transcript": "hi"}])
    provider = AssemblyAIProvider(stt_settings, model=PRO_MODEL)

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect", return_value=cm
    ) as mock_connect:
        await provider.measure_ttft(
            audio_data=b"\x00" * 640,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.01,
        )

    url = mock_connect.call_args.args[0]
    assert "speech_model=universal-1-2-pro" in url
    assert "end_of_turn_confidence_threshold=1.0" in url
    assert "mode=min_latency" in url


@pytest.mark.asyncio
async def test_universal_streaming_multilingual_url_uses_api_speech_model(
    stt_settings: Settings,
) -> None:
    """The multilingual model id maps to its speech_model value with no language lock."""
    cm = fake_connect([{"type": "Turn", "end_of_turn": True, "transcript": "hola"}])
    provider = AssemblyAIProvider(stt_settings, model="universal-streaming-multilingual")

    with patch(
        "coval_bench.providers.stt._streaming.ws_client.connect", return_value=cm
    ) as mock_connect:
        await provider.measure_ttft(
            audio_data=b"\x00" * 640,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.01,
        )

    url = mock_connect.call_args.args[0]
    assert "speech_model=universal-streaming-multilingual" in url
    assert "language_code" not in url
    assert "end_of_turn_confidence_threshold" not in url


# ---------------------------------------------------------------------------
# Invalid model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["bad-model", f"{PRO_MODEL}-x", "universal-pro"])
def test_invalid_model_raises(model: str) -> None:
    with pytest.raises(ValueError, match="Invalid AssemblyAI model"):
        AssemblyAIProvider(make_settings(), model=model)


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
async def test_assemblyai_empty_stream(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = AssemblyAIProvider(stt_settings)

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
async def test_assemblyai_audio_to_final_populated(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """audio_to_final_seconds is set when at least one end_of_turn event is received."""
    events = load_fixture_events("assemblyai", "events-success")
    provider = AssemblyAIProvider(stt_settings)

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
async def test_assemblyai_audio_to_final_none_when_no_end_of_turn(
    stt_settings: Settings, audio_pcm_bytes: bytes
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
    provider = AssemblyAIProvider(stt_settings)

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


@pytest.mark.asyncio
async def test_assemblyai_partial_only_leaves_complete_transcript_none(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """No end_of_turn=true final means no scorable transcript.

    The orchestrator scores WER on any non-null complete_transcript, so an
    unfinalized partial hypothesis must not be promoted to it. Partials are
    still captured (ttft, partial_transcripts) — they just stay out of the
    transcript WER reads.
    """
    events: list[Any] = [
        {"type": "Begin", "id": "test-session", "expires_at": 9999999999},
        {"type": "Turn", "transcript": "hello", "end_of_turn": False},
        {"type": "Turn", "transcript": "hello world", "end_of_turn": False},
    ]
    provider = AssemblyAIProvider(stt_settings)

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

    assert result.complete_transcript is None
    assert result.ttft_seconds is not None
    assert result.partial_transcripts == ["hello", "hello world"]


@pytest.mark.asyncio
async def test_assemblyai_audio_to_final_uses_last_end_of_turn(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """audio_to_final_seconds reflects the LAST end_of_turn event (not the first).

    Verified by using a multi-turn fixture with two end_of_turn events and
    confirming that: (a) audio_to_final_seconds is populated, and (b) the
    complete_transcript contains segments from both turns, proving the receive
    loop did not exit after the first end_of_turn.
    """
    events = load_fixture_events("assemblyai", "events-multi-turn")
    provider = AssemblyAIProvider(stt_settings)

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
    # Both end_of_turn turns must appear — proves we processed all final events
    assert result.complete_transcript is not None
    assert "hello world" in result.complete_transcript
    assert "how are you" in result.complete_transcript


@pytest.mark.asyncio
async def test_assemblyai_multi_turn_complete_transcript(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """All end_of_turn transcripts are joined into complete_transcript (not just first)."""
    events = load_fixture_events("assemblyai", "events-multi-turn")
    provider = AssemblyAIProvider(stt_settings)

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

    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
