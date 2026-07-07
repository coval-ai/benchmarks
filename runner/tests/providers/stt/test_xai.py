# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.xai (XaiSTTProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.xai import XaiSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def _fake_connect(
    events: list[Any],
    *,
    captured: dict[str, Any] | None = None,
) -> Any:
    ws = FakeWebSocket(events)
    if captured is not None:
        captured["ws"] = ws
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_xai_grok_stt_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """grok-stt streams transcript events → TTFT set; transcript.done drives WER."""
    events = load_fixture_events("xai", "events-success")
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert result.first_token_content == "hello"  # noqa: S105 - transcript fixture text
    assert result.complete_transcript == "hello world how are you"
    assert result.audio_to_final_seconds is not None
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_xai_multi_segment_join_empty_done(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """Two distinct speech_final segments + empty done are joined in order.

    Regression guard for the pre-fix bug that scored only the last segment: a
    paused utterance closes each segment with its own speech_final=true partial,
    and the trailing transcript.done is empty.
    """
    events = load_fixture_events("xai", "events-multi-segment")
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_xai_multi_segment_join_trailing_done(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """Speech_final segments plus trailing non-empty transcript.done are all joined."""
    events = load_fixture_events("xai", "events-multi-segment-trailing-done")
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert result.complete_transcript == "hello world how are you thanks"
    assert result.word_count == 6


@pytest.mark.asyncio
async def test_xai_done_echoes_full_transcript_not_duplicated(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """transcript.done echoing the whole utterance must not double the transcript.

    Regression guard for the incident where xAI's transcript.done carried the
    full utterance already closed by speech_final partials instead of the
    unclosed tail. Blindly appending it produced "X X" and ≈100% WER; the dedup
    in _done_already_finalized must drop the echo so the final text stays single.
    """
    events: list[Any] = [
        {"type": "transcript.created", "id": "xai-session-echo"},
        {
            "type": "transcript.partial",
            "text": "hello world",
            "is_final": True,
            "speech_final": True,
        },
        {
            "type": "transcript.partial",
            "text": "how are you",
            "is_final": True,
            "speech_final": True,
        },
        {"type": "transcript.done", "text": "hello world how are you"},
    ]
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_xai_done_echoes_single_segment_not_duplicated(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """Single speech_final segment + done repeating that same segment stays single."""
    events: list[Any] = [
        {"type": "transcript.created", "id": "xai-session-echo-single"},
        {
            "type": "transcript.partial",
            "text": "hello world how are you",
            "is_final": True,
            "speech_final": True,
        },
        {"type": "transcript.done", "text": "hello world how are you"},
    ]
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5


@pytest.mark.asyncio
async def test_xai_done_full_transcript_plus_tail_not_duplicated(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """done carrying the whole utterance plus a new trailing tail must not duplicate.

    transcript.done is the final transcript, so it can repeat the finalized
    segment AND add new words. Appending it whole would double the prefix
    ("hello world hello world thanks"); the merge keeps the authoritative done.
    """
    events: list[Any] = [
        {"type": "transcript.created", "id": "xai-session-full-tail"},
        {
            "type": "transcript.partial",
            "text": "hello world",
            "is_final": True,
            "speech_final": True,
        },
        {"type": "transcript.done", "text": "hello world thanks"},
    ]
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert result.complete_transcript == "hello world thanks"
    assert result.word_count == 3


@pytest.mark.asyncio
async def test_xai_repeated_phrase_tail_is_kept(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """A genuine repeated-phrase tail in done is appended, not dropped.

    Two speech_final "hello world" segments then done "hello world" is a real
    third repetition, not an echo (done is not a prefix of the joined text), so
    it must be kept.
    """
    events: list[Any] = [
        {"type": "transcript.created", "id": "xai-session-repeat"},
        {
            "type": "transcript.partial",
            "text": "hello world",
            "is_final": True,
            "speech_final": True,
        },
        {
            "type": "transcript.partial",
            "text": "hello world",
            "is_final": True,
            "speech_final": True,
        },
        {"type": "transcript.done", "text": "hello world"},
    ]
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert result.complete_transcript == "hello world hello world hello world"


@pytest.mark.asyncio
async def test_xai_interim_cumulative_partials_not_joined(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """Cumulative is_final/speech_final=false partials drive TTFT but are never joined.

    Guards the prior class of bug where cumulative interim partials were
    concatenated into the final transcript. Only the speech_final segment is.
    """
    events: list[Any] = [
        {"type": "transcript.created", "id": "xai-session-cumulative"},
        {"type": "transcript.partial", "text": "hello", "is_final": True, "speech_final": False},
        {
            "type": "transcript.partial",
            "text": "hello world",
            "is_final": True,
            "speech_final": False,
        },
        {
            "type": "transcript.partial",
            "text": "hello world",
            "is_final": True,
            "speech_final": True,
        },
        {"type": "transcript.done", "text": ""},
    ]
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert result.complete_transcript == "hello world"
    assert result.word_count == 2


@pytest.mark.asyncio
async def test_xai_single_speech_final_empty_done(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """A single speech_final segment closes the utterance; empty done adds nothing."""
    events = load_fixture_events("xai", "events-success-empty-done")
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert result.complete_transcript == "hello world how are you"
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_xai_empty_session_errors(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """A session with no partials and an empty transcript.done is an error.

    Models a degraded/dead session (handshake ok, ASR backend emits nothing). It
    must fail — not stamp a spurious audio_to_final from the empty terminal event.
    """
    events = load_fixture_events("xai", "events-empty-session")
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert "empty transcript" in result.error
    assert result.complete_transcript is None
    assert result.ttft_seconds is None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_xai_missing_terminal_errors(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """Stream ends on a chunk is_final with no transcript.done: error, no final metrics."""
    events: list[Any] = [
        {"type": "transcript.created"},
        {"type": "transcript.partial", "text": "hello world", "is_final": True},
    ]
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert "transcript.done" in result.error
    assert result.complete_transcript is None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_xai_websocket_connect_setup(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("xai", "events-success")
    provider = XaiSTTProvider(api_key=fake_api_key)
    captured: dict[str, Any] = {}

    def connect_side_effect(
        url: str,
        additional_headers: dict[str, str] | None = None,
        **_: Any,
    ) -> Any:
        captured["url"] = url
        captured["additional_headers"] = dict(additional_headers or {})
        return _fake_connect(events, captured=captured)

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    parsed = urlparse(captured["url"])
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.x.ai"
    assert parsed.path == "/v1/stt"
    query = parse_qs(parsed.query)
    assert query.get("sample_rate") == ["16000"]
    assert query.get("encoding") == ["pcm"]
    assert query.get("interim_results") == ["true"]
    assert query.get("endpointing") == ["200"]
    assert query.get("language") == ["en"]
    assert query.get("filler_words") == ["true"]
    auth = captured["additional_headers"].get("Authorization")
    assert auth == f"Bearer {fake_api_key.get_secret_value()}"

    sent_binary = [msg for msg in captured["ws"]._sent if isinstance(msg, bytes)]
    sent_json = [json.loads(msg) for msg in captured["ws"]._sent if isinstance(msg, str)]
    assert sent_binary, "expected raw audio frames to be sent"
    assert sent_json, "expected at least one JSON control message to be sent"
    assert sent_json[-1]["type"] == "audio.done"


@pytest.mark.asyncio
async def test_xai_force_finalize_closes_socket(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """After audio.done the gate waits for the final, then closes the socket."""
    events = load_fixture_events("xai", "events-success")
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")
    captured: dict[str, Any] = {}

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
        return_value=_fake_connect(events, captured=captured),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert result.audio_to_final_seconds is not None
    assert captured["ws"]._closed.is_set()


def test_xai_provider_name() -> None:
    provider = XaiSTTProvider(api_key=SecretStr("test"))
    assert provider.name == "xai-grok-stt"


def test_xai_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid xAI STT model"):
        XaiSTTProvider(api_key=SecretStr("k"), model="bad-model")


@pytest.mark.asyncio
async def test_xai_rejects_non_pcm16(fake_api_key: SecretStr) -> None:
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    result = await provider.measure_ttft(
        audio_data=b"\x00" * 16,
        channels=1,
        sample_width=1,
        sample_rate=16000,
    )

    assert result.error is not None
    assert "16-bit PCM" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_xai_rejects_non_positive_realtime_resolution(fake_api_key: SecretStr) -> None:
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    result = await provider.measure_ttft(
        audio_data=b"\x00" * 16,
        channels=1,
        sample_width=2,
        sample_rate=16000,
        realtime_resolution=0,
    )

    assert result.error is not None
    assert "realtime_resolution must be > 0" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_xai_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("xai", "events-error")
    provider = XaiSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert "invalid api key" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_xai_wait_for_ready_deadline(fake_api_key: SecretStr) -> None:
    """_wait_for_ready raises RuntimeError once the deadline elapses without transcript.created."""
    provider = XaiSTTProvider(api_key=fake_api_key)
    ws = FakeWebSocket([b"\x00\x00"])

    times = [0.0, 0.0, 11.0]

    def fake_monotonic() -> float:
        return times.pop(0) if times else 11.0

    with (
        patch(
            "coval_bench.providers.stt.xai.time.monotonic",
            side_effect=fake_monotonic,
        ),
        pytest.raises(RuntimeError, match="xAI did not signal ready"),
    ):
        await provider._wait_for_ready(ws)


@pytest.mark.asyncio
async def test_xai_handshake_closes_before_ready(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """measure_ttft surfaces an error when WS exhausts before transcript.created arrives."""
    events: list[Any] = [b"\x00\x00"]
    provider = XaiSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.xai.ws_client.connect",
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
    assert result.complete_transcript is None
    assert result.ttft_seconds is None
