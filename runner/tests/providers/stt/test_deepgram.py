# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.deepgram (DeepgramProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.deepgram import DeepgramProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_provider(model: str = "nova-3") -> DeepgramProvider:
    return DeepgramProvider(api_key=SecretStr("test-key-deepgram"), model=model)


def _fake_connect(events: list[Any]) -> Any:
    """Return an async context manager that yields a FakeWebSocket."""
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Happy path — nova-3
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_nova3_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("deepgram", "events-success")
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
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
    assert result.complete_transcript is not None
    assert "hello" in result.complete_transcript.lower()
    # VAD: 1 SpeechStarted in fixture
    assert result.vad_events_count == 1
    assert result.vad_first_detected is not None


# ---------------------------------------------------------------------------
# Happy path — default model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_default_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("deepgram", "events-success")
    provider = DeepgramProvider(api_key=fake_api_key, model="default")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
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
# URL builder sanity
# ---------------------------------------------------------------------------


def test_build_websocket_url_nova3() -> None:
    p = make_provider("nova-3")
    url = p._build_websocket_url(16000, 1)
    assert "nova-3" in url
    assert "api.deepgram.com" in url
    # Native silence endpointing disabled; we force the final via Finalize (TTFS parity).
    assert "endpointing=false" in url
    assert "filler_words=true" in url


def test_build_websocket_url_flux() -> None:
    p = make_provider("flux-general-en")
    url = p._build_websocket_url(16000, 1)
    assert url.startswith("wss://api.deepgram.com/v2/listen")
    assert "preview" not in url
    assert "flux-general-en" in url
    # v2/listen rejects v1-only query params as unknown and closes the WS upgrade
    # with HTTP 400 — interim_results, no_delay AND channels must all be absent.
    assert "interim_results" not in url
    assert "no_delay" not in url
    assert "channels" not in url
    # endpointing is a v1-only param and Flux can't be forced anyway.
    assert "endpointing" not in url
    # filler_words is a v1-only param; Flux 400s on it.
    assert "filler_words" not in url


@pytest.mark.asyncio
async def test_nova_sends_finalize_before_close(
    fake_api_key: SecretStr, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("coval_bench.providers.stt.deepgram._FINAL_WAIT_S", 0.05)
    sent: list[Any] = []
    final = {
        "type": "Results",
        "is_final": True,
        "channel": {"alternatives": [{"transcript": "hello"}]},
    }
    ws = FakeWebSocket([final], on_send=sent.append)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch("coval_bench.providers.stt.deepgram.ws_client.connect", return_value=cm):
        result = await provider.measure_ttft(
            audio_data=b"\x00" * 640,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.01,
        )

    text = [m for m in sent if isinstance(m, str)]
    assert '"Finalize"' in text[-2]
    assert '"CloseStream"' in text[-1]
    # The forced final is captured before the close (gate + response-capture path).
    assert result.audio_to_final_seconds is not None


@pytest.mark.asyncio
async def test_flux_does_not_send_finalize(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    sent: list[Any] = []
    ws = FakeWebSocket([], on_send=sent.append)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    provider = DeepgramProvider(api_key=fake_api_key, model="flux-general-en")

    with patch("coval_bench.providers.stt.deepgram.ws_client.connect", return_value=cm):
        await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.01,
        )

    text = [m for m in sent if isinstance(m, str)]
    assert not any("Finalize" in m for m in text)
    assert '"CloseStream"' in text[-1]


def test_build_websocket_url_flux_rejects_non_mono() -> None:
    p = make_provider("flux-general-en")
    with pytest.raises(ValueError, match="Flux models require mono audio"):
        p._build_websocket_url(16000, 2)


def test_build_websocket_url_flux_multi() -> None:
    p = make_provider("flux-general-multi")
    url = p._build_websocket_url(16000, 1)
    assert url.startswith("wss://api.deepgram.com/v2/listen")
    assert "preview" not in url
    assert "model=flux-general-multi" in url
    # No language hint — multilingual model auto-detects.
    assert "language=" not in url
    assert "interim_results" not in url
    assert "no_delay" not in url
    assert "channels" not in url


# ---------------------------------------------------------------------------
# Invalid model
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Deepgram model"):
        DeepgramProvider(api_key=SecretStr("k"), model="nova-99")


# ---------------------------------------------------------------------------
# Failure path — empty event stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
        return_value=_fake_connect([]),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    # No transcript events → no complete transcript
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


# ---------------------------------------------------------------------------
# Failure path — error event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("deepgram", "events-error")
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    # Error events don't produce a transcript
    assert result.complete_transcript is None


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


def test_provider_name_with_model() -> None:
    p = make_provider("nova-2")
    assert p.name == "deepgram-nova-2"


def test_provider_name_default() -> None:
    p = make_provider("default")
    assert p.name == "deepgram"


# ---------------------------------------------------------------------------
# audio_to_final_seconds — nova-3 (speech_final events)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_audio_to_final_populated(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """audio_to_final_seconds is set when at least one speech_final event is received."""
    events = load_fixture_events("deepgram", "events-success")
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
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
async def test_deepgram_audio_to_final_none_when_no_final(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """audio_to_final_seconds is None when no speech_final transcript is received."""
    # Events with only non-final results
    events: list[Any] = [
        {"type": "Connected", "session_id": "test-no-final"},
        {
            "type": "Results",
            "is_final": False,
            "speech_final": False,
            "channel": {
                "alternatives": [
                    {
                        "transcript": "hello",
                        "words": [{"word": "hello", "punctuated_word": "hello"}],
                    }
                ]
            },
        },
    ]
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
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
async def test_deepgram_audio_to_final_uses_last_final(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """audio_to_final_seconds reflects the LAST speech_final event.

    Verified by using a multi-final fixture with two speech_final events and
    confirming that: (a) audio_to_final_seconds is populated, and (b) the
    complete_transcript contains segments from both final events, proving the
    receive loop did not exit after the first speech_final.
    """
    events = load_fixture_events("deepgram", "events-multi-final")
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
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
    # Both speech_final segments must appear — proves we processed all final events
    assert result.complete_transcript is not None
    complete_lower = result.complete_transcript.lower()
    assert "hello world" in complete_lower
    assert "how are you" in complete_lower


# ---------------------------------------------------------------------------
# audio_to_final_seconds — default model (is_final events)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_default_audio_to_final(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """audio_to_final_seconds is set for default model using is_final events."""
    events: list[Any] = [
        {"type": "Connected", "session_id": "test-default"},
        {
            "type": "Results",
            "is_final": True,
            "speech_final": False,
            "channel": {
                "alternatives": [
                    {
                        "transcript": "hello world",
                        "words": [
                            {"word": "hello", "punctuated_word": "Hello"},
                            {"word": "world", "punctuated_word": "world"},
                        ],
                    }
                ]
            },
        },
    ]
    provider = DeepgramProvider(api_key=fake_api_key, model="default")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
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


# ---------------------------------------------------------------------------
# audio_to_final_seconds + TTFT — flux-general-en
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_flux_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """flux-general-en: TTFT and audio_to_final_seconds are populated from TurnInfo messages."""
    events = load_fixture_events("deepgram", "events-flux-success")
    provider = DeepgramProvider(api_key=fake_api_key, model="flux-general-en")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
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
    assert result.audio_to_final_seconds is not None
    assert result.audio_to_final_seconds >= 0


@pytest.mark.asyncio
async def test_deepgram_flux_concatenates_multiple_turns(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """flux: turns are concatenated by turn_index, not collapsed to the last turn.

    The single-turn happy-path fixture can't distinguish concatenation from
    last-turn-wins. This drives two turns (delivered out of order) so a revert to
    keeping only the latest turn — or a switch that drops turn ordering — fails.
    """
    events = [
        {"type": "Connected", "session_id": "test-flux-multi"},
        {"type": "TurnInfo", "event": "StartOfTurn", "turn_index": 1, "transcript": "how"},
        {"type": "TurnInfo", "event": "EndOfTurn", "turn_index": 1, "transcript": "how are you"},
        {"type": "TurnInfo", "event": "StartOfTurn", "turn_index": 0, "transcript": "hello"},
        {"type": "TurnInfo", "event": "EndOfTurn", "turn_index": 0, "transcript": "hello world"},
    ]
    provider = DeepgramProvider(api_key=fake_api_key, model="flux-general-en")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    # Ordered by turn_index regardless of arrival order; both turns present.
    assert result.complete_transcript == "hello world how are you"


@pytest.mark.asyncio
async def test_deepgram_flux_audio_to_final_none_when_no_transcript(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """flux-general-en: audio_to_final_seconds is None when no transcript events arrive."""
    provider = DeepgramProvider(api_key=fake_api_key, model="flux-general-en")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
        return_value=_fake_connect([]),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.audio_to_final_seconds is None


# ---------------------------------------------------------------------------
# Happy path — flux-general-multi
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_flux_multi_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """flux-general-multi: TTFT and complete_transcript are populated from TurnInfo messages."""
    events = load_fixture_events("deepgram", "events-flux-success")
    provider = DeepgramProvider(api_key=fake_api_key, model="flux-general-multi")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
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
    assert result.complete_transcript == "hello world how are you"
    assert result.audio_to_final_seconds is not None
    assert result.audio_to_final_seconds >= 0


# ---------------------------------------------------------------------------
# Transcript assembly — keep ALL is_final segments, not just speech_final
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepgram_keeps_isfinal_only_segments(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """nova-3: an is_final segment without speech_final must not be dropped.

    Deepgram chunks long utterances into multiple is_final pieces, with
    speech_final only on the last one before a pause. Keying assembly on
    speech_final dropped the in-between pieces; we accumulate on is_final.
    """
    events: list[Any] = [
        {"type": "Connected", "session_id": "test-isfinal"},
        {
            "type": "Results",
            "is_final": True,
            "speech_final": False,  # finalized piece, but not an endpoint
            "channel": {"alternatives": [{"transcript": "the quick brown fox"}]},
        },
        {
            "type": "Results",
            "is_final": True,
            "speech_final": True,  # endpoint — only this survived before the fix
            "channel": {"alternatives": [{"transcript": "jumps over the lazy dog"}]},
        },
    ]
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.complete_transcript == "the quick brown fox jumps over the lazy dog"


@pytest.mark.asyncio
async def test_deepgram_includes_unfinalized_tail(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """nova-3: a tail the stream closes before finalizing is still included.

    After the last is_final, interim results keep arriving but never finalize.
    The longest such interim is appended so the trailing words aren't lost.
    """
    events: list[Any] = [
        {"type": "Connected", "session_id": "test-tail"},
        {
            "type": "Results",
            "is_final": True,
            "speech_final": True,
            "channel": {"alternatives": [{"transcript": "first sentence"}]},
        },
        {
            "type": "Results",
            "is_final": False,
            "speech_final": False,
            "channel": {"alternatives": [{"transcript": "second"}]},
        },
        {
            "type": "Results",
            "is_final": False,
            "speech_final": False,
            "channel": {"alternatives": [{"transcript": "second sentence tail"}]},
        },
    ]
    provider = DeepgramProvider(api_key=fake_api_key, model="nova-3")

    with patch(
        "coval_bench.providers.stt.deepgram.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.complete_transcript == "first sentence second sentence tail"
