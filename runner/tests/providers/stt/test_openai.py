# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.openai (OpenAISTTProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.openai import OpenAISTTProvider
from tests.providers.stt.conftest import fake_connect, load_fixture_events, make_settings

_CONNECT = "coval_bench.providers.stt._streaming.ws_client.connect"


async def _measure(provider: OpenAISTTProvider, audio: bytes, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "audio_data": audio,
        "channels": 1,
        "sample_width": 2,
        "sample_rate": 16000,
        "realtime_resolution": 0.5,
    }
    kwargs.update(overrides)
    return await provider.measure_ttft(**kwargs)


def _cm(ws: Any) -> Any:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_openai_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(stt_settings)

    with patch(
        _CONNECT, return_value=fake_connect(load_fixture_events("openai", "events-success"))
    ):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.audio_to_final_seconds is not None
    assert result.complete_transcript == "hello world how are you"
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_openai_single_connection_and_session_shape(audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(make_settings(openai_api_key=SecretStr("test-key-openai")))
    connections: list[dict[str, Any]] = []

    def connect_side_effect(
        url: str,
        additional_headers: dict[str, str] | None = None,
        **_: Any,
    ) -> Any:
        cm = fake_connect(load_fixture_events("openai", "events-success"))
        connections.append(
            {"url": url, "additional_headers": dict(additional_headers or {}), "ws": cm.ws}
        )
        return cm

    with patch(_CONNECT, side_effect=connect_side_effect):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    assert len(connections) == 1

    conn = connections[0]
    parsed = urlparse(conn["url"])
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.openai.com"
    assert parsed.path == "/v1/realtime"
    assert parse_qs(parsed.query).get("intent") == ["transcription"]
    assert conn["additional_headers"].get("Authorization") == "Bearer test-key-openai"
    assert "OpenAI-Beta" not in conn["additional_headers"]

    sent = [json.loads(m) for m in conn["ws"]._sent if isinstance(m, str)]
    session_update = sent[0]
    assert session_update["type"] == "session.update"
    assert session_update["session"]["audio"]["input"]["turn_detection"] is None
    assert sent[-1]["type"] == "input_audio_buffer.commit"

    appends = [m for m in sent if m["type"] == "input_audio_buffer.append"]
    assert len(appends) > 0
    first_chunk = base64.b64decode(appends[0]["audio"])
    assert len(first_chunk) == 24000  # 0.5s × 24 kHz × 2 bytes (resampled from 16 kHz)


@pytest.mark.asyncio
async def test_openai_delta_partials_captured(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    provider = OpenAISTTProvider(stt_settings)

    with patch(
        _CONNECT, return_value=fake_connect(load_fixture_events("openai", "events-success"))
    ):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.first_token_content is not None
    assert result.first_token_content.startswith("hello")
    assert result.partial_transcripts is not None
    assert len(result.partial_transcripts) >= 2
    assert any("hello world" in p for p in result.partial_transcripts)


@pytest.mark.asyncio
async def test_openai_delta_ttft_not_overwritten_by_completed(
    stt_settings: Settings,
    audio_pcm_bytes: bytes,
) -> None:
    """TTFT is set at the first delta; completed must not replace first_token_content."""
    events = [
        {"type": "session.created", "session": {"id": "sess_001"}},
        {
            "type": "session.updated",
            "session": {"type": "transcription", "audio": {"input": {"turn_detection": None}}},
        },
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item_001",
            "content_index": 0,
            "delta": "hello",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_001",
            "content_index": 0,
            "transcript": "hello world",
        },
    ]
    provider = OpenAISTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    assert result.first_token_content == "hello"  # noqa: S105
    assert result.complete_transcript == "hello world"
    assert result.ttft_seconds is not None
    if result.audio_to_final_seconds is not None:
        assert result.ttft_seconds <= result.audio_to_final_seconds


class _SendFailsRecvBlocksWebSocket:
    """Handshake succeeds, the first audio append fails, and the receiver never ends."""

    def __init__(self, ready_events: list[dict[str, Any]]) -> None:
        self._ready_events = list(ready_events)
        self._never = asyncio.Event()
        self._sent: list[Any] = []

    async def __aenter__(self) -> _SendFailsRecvBlocksWebSocket:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def send(self, msg: bytes | str) -> None:
        self._sent.append(msg)
        if isinstance(msg, str) and "input_audio_buffer.append" in msg:
            raise ConnectionResetError("simulated send failure")

    async def recv(self) -> str:
        if self._ready_events:
            return json.dumps(self._ready_events.pop(0))
        await self._never.wait()  # block until cancelled
        raise AssertionError("unreachable")

    def __aiter__(self) -> _SendFailsRecvBlocksWebSocket:
        return self

    async def __anext__(self) -> str:
        await self._never.wait()  # receiver blocks forever until cancelled
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_openai_send_failure_propagates_without_hang(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """A sender error surfaces immediately; the blocked receiver is cancelled, not awaited."""
    provider = OpenAISTTProvider(stt_settings)
    ws = _SendFailsRecvBlocksWebSocket(
        [{"type": "session.updated", "session": {"type": "transcription"}}]
    )

    with patch(_CONNECT, return_value=_cm(ws)):
        result = await asyncio.wait_for(_measure(provider, audio_pcm_bytes), timeout=2.0)

    assert result.error is not None
    assert "simulated send failure" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_in_stream_error_event(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    """Regression guard: a mid-stream top-level ``error`` event becomes result.error."""
    events = [
        {"type": "session.updated", "session": {"type": "transcription"}},
        {"type": "error", "error": {"message": "rate limit exceeded"}},
    ]
    provider = OpenAISTTProvider(stt_settings)
    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert "rate limit exceeded" in result.error
    assert result.complete_transcript is None


class _RecvFailsSendBlocksWebSocket:
    """Handshake succeeds, the receiver reports a failure, and the sender never ends."""

    def __init__(self, ready_events: list[dict[str, Any]], fail_event: dict[str, Any]) -> None:
        self._ready_events = list(ready_events)
        self._fail_events = [fail_event]
        self._never = asyncio.Event()
        self._sent: list[Any] = []

    async def __aenter__(self) -> _RecvFailsSendBlocksWebSocket:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def send(self, msg: bytes | str) -> None:
        self._sent.append(msg)
        if isinstance(msg, str) and "input_audio_buffer.append" in msg:
            await self._never.wait()  # sender blocks until cancelled

    async def recv(self) -> str:
        if self._ready_events:
            return json.dumps(self._ready_events.pop(0))
        await self._never.wait()
        raise AssertionError("unreachable")

    def __aiter__(self) -> _RecvFailsSendBlocksWebSocket:
        return self

    async def __anext__(self) -> str:
        if self._fail_events:
            return json.dumps(self._fail_events.pop(0))
        await self._never.wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_openai_transcription_failed_aborts_sender_without_hang(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """A failed event cancels the still-running sender instead of awaiting the full clip."""
    provider = OpenAISTTProvider(stt_settings)
    ws = _RecvFailsSendBlocksWebSocket(
        [{"type": "session.updated", "session": {"type": "transcription"}}],
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "error": {"message": "audio too short"},
        },
    )

    with patch(_CONNECT, return_value=_cm(ws)):
        result = await asyncio.wait_for(_measure(provider, audio_pcm_bytes), timeout=2.0)

    assert result.error is not None
    assert "audio too short" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_session_ready_timeout_fires(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """The handshake deadline fires when the server never sends session.updated."""
    provider = OpenAISTTProvider(stt_settings)
    ws = _SendFailsRecvBlocksWebSocket([])  # recv() blocks; no session.updated ever

    with (
        patch("coval_bench.providers.stt.openai._READY_TIMEOUT_S", 0.05),
        patch(_CONNECT, return_value=_cm(ws)),
    ):
        result = await asyncio.wait_for(_measure(provider, audio_pcm_bytes), timeout=2.0)

    assert result.error is not None
    assert "session.updated" in result.error


@pytest.mark.parametrize(
    "model",
    ["gpt-realtime-whisper", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
)
def test_openai_valid_models_accepted(model: str) -> None:
    provider = OpenAISTTProvider(make_settings(), model=model)
    assert provider.name == f"openai-{model}"
    assert provider.model == model


def test_openai_default_model_is_whisper() -> None:
    assert OpenAISTTProvider(make_settings()).model == "gpt-realtime-whisper"


def test_openai_requires_api_key() -> None:
    with pytest.raises(ValueError, match="openai_api_key is required"):
        OpenAISTTProvider(make_settings(openai_api_key=None))


@pytest.mark.asyncio
async def test_openai_session_update_carries_model(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """The constructed model is sent in the session.update transcription config."""
    provider = OpenAISTTProvider(stt_settings, model="gpt-4o-transcribe")
    cm = fake_connect(load_fixture_events("openai", "events-success"))

    with patch(_CONNECT, return_value=cm):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    sent = [json.loads(m) for m in cm.ws._sent if isinstance(m, str)]
    transcription = sent[0]["session"]["audio"]["input"]["transcription"]
    assert transcription["model"] == "gpt-4o-transcribe"


@pytest.mark.asyncio
async def test_openai_rejects_non_positive_realtime_resolution(stt_settings: Settings) -> None:
    provider = OpenAISTTProvider(stt_settings)

    result = await _measure(provider, b"\x00\x00", realtime_resolution=0)

    assert result.error is not None
    assert "realtime_resolution must be > 0" in result.error


@pytest.mark.asyncio
async def test_openai_session_ready_stream_ended(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    """Handshake stream ends before session.updated → bounded error (StopAsyncIteration path)."""
    provider = OpenAISTTProvider(stt_settings)

    with (
        patch("coval_bench.providers.stt.openai._READY_TIMEOUT_S", 0.05),
        patch(
            _CONNECT,
            return_value=fake_connect(
                [{"type": "session.created", "session": {"id": "sess_slow"}}]
            ),
        ),
    ):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert "session.updated" in result.error


@pytest.mark.asyncio
async def test_openai_rejects_non_mono_input(
    stt_settings: Settings,
    audio_pcm_bytes: bytes,
) -> None:
    provider = OpenAISTTProvider(stt_settings)

    result = await _measure(provider, audio_pcm_bytes, channels=2)

    assert result.error is not None
    assert "mono 16-bit PCM" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_rejects_non_pcm16_input(
    stt_settings: Settings,
    audio_pcm_bytes: bytes,
) -> None:
    provider = OpenAISTTProvider(stt_settings)

    result = await _measure(provider, audio_pcm_bytes, sample_width=1)

    assert result.error is not None
    assert "16-bit PCM" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_session_setup_error(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(stt_settings)
    events = [
        {
            "type": "error",
            "error": {"message": "Turn detection is not supported for this transcription model."},
        }
    ]

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert "session setup" in result.error
    assert "Turn detection is not supported" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_error_event(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(load_fixture_events("openai", "events-error"))):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert "quota exceeded" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_no_transcript(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(stt_settings)

    with patch(
        _CONNECT, return_value=fake_connect(load_fixture_events("openai", "events-no-transcript"))
    ):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error == "ws_closed_without_completed"
    assert result.ttft_seconds is not None
    assert result.complete_transcript is not None
    assert "hello" in result.complete_transcript


@pytest.mark.asyncio
async def test_openai_item_id_tracking(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(stt_settings)

    with patch(
        _CONNECT, return_value=fake_connect(load_fixture_events("openai", "events-multi-item"))
    ):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    # `.completed` is scoped to its own item_id (no cross-item merge), so item_001's
    # final is only its own deltas; item_002's text stays in the partials.
    assert result.complete_transcript == "hello how"
    assert result.first_token_content is not None
    assert result.first_token_content.startswith("hello")
    assert result.partial_transcripts is not None
    assert any("world are you" in p for p in result.partial_transcripts)
