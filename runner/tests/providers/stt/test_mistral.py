# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.mistral (MistralSTTProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.mistral import MistralSTTProvider
from tests.providers.stt.conftest import fake_connect, load_fixture_events, make_settings

_CONNECT = "coval_bench.providers.stt._streaming.ws_client.connect"
_CREATED = {"type": "session.created", "session": {"request_id": "r1", "audio_format": None}}


def _done(text: str) -> dict[str, Any]:
    return {
        "type": "transcription.done",
        "model": "voxtral-mini-transcribe-realtime-2602",
        "text": text,
        "language": "en" if text else None,
        "usage": {"prompt_tokens": 0, "completion_tokens": len(text.split()), "total_tokens": 0},
    }


async def _measure(provider: MistralSTTProvider, audio: bytes, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "audio_data": audio,
        "channels": 1,
        "sample_width": 2,
        "sample_rate": 16000,
        "realtime_resolution": 0.5,
    }
    kwargs.update(overrides)
    return await provider.measure_ttft(**kwargs)


class _SilentWebSocket:
    """The handshake never completes: recv() blocks until cancelled."""

    async def __aenter__(self) -> _SilentWebSocket:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def recv(self) -> str:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_mistral_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """Deltas drive TTFT; transcription.done carries the final transcript."""
    events = load_fixture_events("mistral", "events-success")
    provider = MistralSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content == "hello"  # noqa: S105 - transcript fixture text
    assert result.complete_transcript == "hello world how are you"
    assert result.audio_to_final_seconds is not None
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_mistral_missing_done_falls_back_to_deltas(
    stt_settings: Settings,
    audio_pcm_bytes: bytes,
) -> None:
    """Stream closing without transcription.done keeps the joined deltas but no final time."""
    events: list[Any] = [
        _CREATED,
        {"type": "transcription.text.delta", "text": "hello"},
        {"type": "transcription.text.delta", "text": " world"},
    ]
    provider = MistralSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.complete_transcript == "hello world"
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_mistral_empty_deltas_ignored_for_ttft(
    stt_settings: Settings,
    audio_pcm_bytes: bytes,
) -> None:
    """Whitespace-only deltas never stamp TTFT or first_token_content."""
    events: list[Any] = [
        _CREATED,
        {"type": "transcription.text.delta", "text": " "},
        {"type": "transcription.text.delta", "text": "hi"},
        _done("hi"),
    ]
    provider = MistralSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    assert result.first_token_content == "hi"  # noqa: S105 - transcript fixture text
    assert result.complete_transcript == "hi"


@pytest.mark.asyncio
async def test_mistral_space_deltas_reach_fallback_transcript(
    stt_settings: Settings,
    audio_pcm_bytes: bytes,
) -> None:
    """A standalone " " delta is a word boundary in the joined fallback transcript."""
    events: list[Any] = [
        _CREATED,
        {"type": "transcription.text.delta", "text": "hello"},
        {"type": "transcription.text.delta", "text": " "},
        {"type": "transcription.text.delta", "text": "world"},
    ]
    provider = MistralSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.complete_transcript == "hello world"
    assert result.partial_transcripts == ["hello", "world"]


@pytest.mark.asyncio
async def test_mistral_partial_transcripts_accumulate(
    stt_settings: Settings,
    audio_pcm_bytes: bytes,
) -> None:
    """Each non-empty delta is recorded in partial_transcripts in arrival order."""
    events = load_fixture_events("mistral", "events-success")
    provider = MistralSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    assert result.partial_transcripts == ["hello", " world", " how are you"]


@pytest.mark.asyncio
async def test_mistral_first_done_wins(
    stt_settings: Settings,
    audio_pcm_bytes: bytes,
) -> None:
    """A rogue early transcription.done ends the session; later events are ignored.

    transcription.done is session-terminal in the protocol (the vendor SDK also
    stops on the first one), so a mid-utterance done must not be double-counted
    or merged with anything the server sends afterwards.
    """
    events: list[Any] = [
        _CREATED,
        {"type": "transcription.text.delta", "text": "hello world"},
        _done("hello world"),
        {"type": "transcription.text.delta", "text": " how are you"},
        _done("hello world how are you"),
    ]
    provider = MistralSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    assert result.complete_transcript == "hello world"
    assert result.partial_transcripts == ["hello world"]
    assert result.audio_to_final_seconds is not None


@pytest.mark.asyncio
async def test_mistral_websocket_connect_setup(audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("mistral", "events-success")
    provider = MistralSTTProvider(make_settings(mistral_api_key=SecretStr("test-key-mistral")))
    captured: dict[str, Any] = {}

    def connect_side_effect(
        url: str,
        additional_headers: dict[str, str] | None = None,
        **_: Any,
    ) -> Any:
        cm = fake_connect(events)
        captured.update(url=url, additional_headers=dict(additional_headers or {}), ws=cm.ws)
        return cm

    with patch(_CONNECT, side_effect=connect_side_effect):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    parsed = urlparse(captured["url"])
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.mistral.ai"
    assert parsed.path == "/v1/audio/transcriptions/realtime"
    query = parse_qs(parsed.query)
    assert query.get("model") == ["voxtral-mini-transcribe-realtime-2602"]
    assert captured["additional_headers"].get("Authorization") == "Bearer test-key-mistral"

    sent_json = [json.loads(msg) for msg in captured["ws"]._sent if isinstance(msg, str)]
    assert sent_json[0]["type"] == "session.update"
    assert sent_json[0]["session"]["audio_format"] == {
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
    }
    appends = [msg for msg in sent_json if msg["type"] == "input_audio.append"]
    assert appends, "expected input_audio.append frames to be sent"
    sent_audio = b"".join(base64.b64decode(msg["audio"]) for msg in appends)
    assert sent_audio == audio_pcm_bytes
    assert sent_json[-2]["type"] == "input_audio.flush"
    assert sent_json[-1]["type"] == "input_audio.end"


@pytest.mark.asyncio
async def test_mistral_empty_session_errors(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """No deltas and an empty transcription.done is an error, not a final.

    Models a degraded/dead session (handshake ok, ASR backend emits nothing). It
    must fail — not stamp a spurious audio_to_final from the empty terminal event.
    """
    events: list[Any] = [_CREATED, _done("")]
    provider = MistralSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert "empty transcript" in result.error
    assert result.complete_transcript is None
    assert result.ttft_seconds is None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_mistral_error_event(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("mistral", "events-error")
    provider = MistralSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert "invalid api key" in result.error
    assert result.complete_transcript is None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_mistral_handshake_error(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """An error event before session.created fails the run."""
    events: list[Any] = [
        {"type": "error", "error": {"code": "unauthorized", "message": "invalid api key"}},
    ]
    provider = MistralSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert "invalid api key" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_mistral_handshake_closes_before_created(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """WS exhausting before session.created surfaces an error."""
    events: list[Any] = [b"\x00\x00"]
    provider = MistralSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert "session.created" in result.error
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_mistral_session_created_deadline(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """The handshake deadline fires when session.created never arrives."""
    provider = MistralSTTProvider(stt_settings)

    with (
        patch("coval_bench.providers.stt.mistral._SESSION_CREATED_TIMEOUT_S", 0.05),
        patch(_CONNECT, return_value=_SilentWebSocket()),
    ):
        result = await asyncio.wait_for(_measure(provider, audio_pcm_bytes), timeout=2.0)

    assert result.error is not None
    assert "session.created" in result.error
    assert result.ttft_seconds is None


def test_mistral_provider_name() -> None:
    provider = MistralSTTProvider(make_settings())
    assert provider.name == "mistral-voxtral-mini-transcribe-realtime-2602"


def test_mistral_requires_api_key() -> None:
    with pytest.raises(ValueError, match="mistral_api_key is required"):
        MistralSTTProvider(make_settings(mistral_api_key=None))


@pytest.mark.asyncio
async def test_mistral_rejects_stereo(stt_settings: Settings) -> None:
    provider = MistralSTTProvider(stt_settings)

    result = await _measure(provider, b"\x00" * 16, channels=2)

    assert result.error is not None
    assert "mono" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_mistral_rejects_non_pcm16(stt_settings: Settings) -> None:
    provider = MistralSTTProvider(stt_settings)

    result = await _measure(provider, b"\x00" * 16, sample_width=1)

    assert result.error is not None
    assert "16-bit PCM" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_mistral_rejects_unsupported_sample_rate(stt_settings: Settings) -> None:
    provider = MistralSTTProvider(stt_settings)

    result = await _measure(provider, b"\x00" * 16, sample_rate=12345)

    assert result.error is not None
    assert "12345" in result.error
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_mistral_rejects_non_positive_realtime_resolution(stt_settings: Settings) -> None:
    provider = MistralSTTProvider(stt_settings)

    result = await _measure(provider, b"\x00" * 16, realtime_resolution=0)

    assert result.error is not None
    assert "realtime_resolution must be > 0" in result.error
    assert result.ttft_seconds is None
