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
async def test_universal_streaming_terminates_without_force_endpoint(
    fake_api_key: SecretStr,
) -> None:
    """TTFS-excluded models end the session per the vendor's streaming-WER guide."""
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
    assert "end_of_turn_confidence_threshold" not in url

    text = [m for m in sent if isinstance(m, str)]
    assert text == ['{"type": "Terminate"}']
    assert result.complete_transcript == "hello world"


@pytest.mark.asyncio
async def test_universal_3_5_pro_force_endpoint_before_terminate(
    fake_api_key: SecretStr,
) -> None:
    sent: list[Any] = []
    final = {"type": "Turn", "end_of_turn": True, "transcript": "hello world"}
    ws = FakeWebSocket([final], on_send=sent.append)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    provider = AssemblyAIProvider(api_key=fake_api_key, model="universal-3.5-pro")

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
    assert result.audio_to_final_seconds is not None


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    p = make_provider()
    assert p.name == "assemblyai-universal-streaming"


def test_provider_name_universal_3_5_pro() -> None:
    p = AssemblyAIProvider(api_key=SecretStr("k"), model="universal-3.5-pro")
    assert p.name == "assemblyai-universal-3.5-pro"


def test_provider_name_universal_streaming_multilingual() -> None:
    p = AssemblyAIProvider(api_key=SecretStr("k"), model="universal-streaming-multilingual")
    assert p.name == "assemblyai-universal-streaming-multilingual"


@pytest.mark.asyncio
async def test_universal_3_5_pro_url_uses_api_speech_model(fake_api_key: SecretStr) -> None:
    """The friendly model id maps to the API's speech_model value on the wire."""
    ws = FakeWebSocket([{"type": "Turn", "end_of_turn": True, "transcript": "hi"}])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    provider = AssemblyAIProvider(api_key=fake_api_key, model="universal-3.5-pro")

    with patch(
        "coval_bench.providers.stt.assemblyai.ws_client.connect", return_value=cm
    ) as mock_connect:
        await provider.measure_ttft(
            audio_data=b"\x00" * 640,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.01,
        )

    url = mock_connect.call_args.args[0]
    assert "speech_model=universal-3-5-pro" in url
    assert "end_of_turn_confidence_threshold=1.0" in url


@pytest.mark.asyncio
async def test_universal_3_5_pro_url_has_voice_agent_config(fake_api_key: SecretStr) -> None:
    """universal-3.5-pro connects with the vendor's min_latency voice-agent preset."""
    ws = FakeWebSocket([{"type": "Turn", "end_of_turn": True, "transcript": "hi"}])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    provider = AssemblyAIProvider(api_key=fake_api_key, model="universal-3.5-pro")

    with patch(
        "coval_bench.providers.stt.assemblyai.ws_client.connect", return_value=cm
    ) as mock_connect:
        await provider.measure_ttft(
            audio_data=b"\x00" * 640,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.01,
        )

    url = mock_connect.call_args.args[0]
    assert "mode=min_latency" in url

    # Other models keep the stock configuration
    from coval_bench.providers.stt.assemblyai import _MODEL_EXTRA_PARAMS

    assert set(_MODEL_EXTRA_PARAMS) == {"universal-3.5-pro"}


@pytest.mark.asyncio
async def test_universal_streaming_multilingual_url_uses_api_speech_model(
    fake_api_key: SecretStr,
) -> None:
    """The multilingual model id maps to its speech_model value with no language lock."""
    ws = FakeWebSocket([{"type": "Turn", "end_of_turn": True, "transcript": "hola"}])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    provider = AssemblyAIProvider(api_key=fake_api_key, model="universal-streaming-multilingual")

    with patch(
        "coval_bench.providers.stt.assemblyai.ws_client.connect", return_value=cm
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
async def test_assemblyai_partial_only_leaves_complete_transcript_none(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
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

    assert result.complete_transcript is None
    assert result.ttft_seconds is not None
    assert result.partial_transcripts == ["hello", "hello world"]


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
