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

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.openai import OpenAISTTProvider
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
async def test_openai_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.openai.ws_client.connect",
        return_value=_fake_connect(load_fixture_events("openai", "events-success")),
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
    assert result.audio_to_final_seconds is not None
    assert result.complete_transcript == "hello world how are you"
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_openai_single_connection_and_session_shape(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = OpenAISTTProvider(api_key=fake_api_key)
    connections: list[dict[str, Any]] = []

    def connect_side_effect(
        url: str,
        additional_headers: dict[str, str] | None = None,
        **_: Any,
    ) -> Any:
        captured: dict[str, Any] = {
            "url": url,
            "additional_headers": dict(additional_headers or {}),
        }
        cm = _fake_connect(load_fixture_events("openai", "events-success"), captured=captured)
        connections.append(captured)
        return cm

    with patch(
        "coval_bench.providers.stt.openai.ws_client.connect",
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
    assert len(connections) == 1

    conn = connections[0]
    parsed = urlparse(conn["url"])
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.openai.com"
    assert parsed.path == "/v1/realtime"
    assert parse_qs(parsed.query).get("intent") == ["transcription"]
    assert (
        conn["additional_headers"].get("Authorization")
        == f"Bearer {fake_api_key.get_secret_value()}"
    )
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
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = OpenAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.openai.ws_client.connect",
        return_value=_fake_connect(load_fixture_events("openai", "events-success")),
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
    assert result.first_token_content is not None
    assert result.first_token_content.startswith("hello")
    assert result.partial_transcripts is not None
    assert len(result.partial_transcripts) >= 2
    assert any("hello world" in p for p in result.partial_transcripts)


@pytest.mark.asyncio
async def test_openai_delta_ttft_not_overwritten_by_completed(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """TTFT is set at the first delta; completed must not replace first_token_content.

    Both code paths in _receive call set_first_token. A regression (missing guard)
    would set first_token_content from the completed event instead of the delta.
    """
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
    provider = OpenAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.openai.ws_client.connect",
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
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A sender error surfaces immediately; the blocked receiver is cancelled, not awaited."""
    provider = OpenAISTTProvider(api_key=fake_api_key)
    ws = _SendFailsRecvBlocksWebSocket(
        [{"type": "session.updated", "session": {"type": "transcription"}}]
    )
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("coval_bench.providers.stt.openai.ws_client.connect", return_value=cm):
        result = await asyncio.wait_for(
            provider.measure_ttft(
                audio_data=audio_pcm_bytes,
                channels=1,
                sample_width=2,
                sample_rate=16000,
                realtime_resolution=0.5,
            ),
            timeout=2.0,
        )

    assert result.error is not None
    assert "simulated send failure" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_in_stream_error_event(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """Regression guard: a mid-stream top-level ``error`` event becomes result.error."""
    events = [
        {"type": "session.updated", "session": {"type": "transcription"}},
        {"type": "error", "error": {"message": "rate limit exceeded"}},
    ]
    provider = OpenAISTTProvider(api_key=fake_api_key)
    with patch(
        "coval_bench.providers.stt.openai.ws_client.connect",
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
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A failed event cancels the still-running sender instead of awaiting the full clip."""
    provider = OpenAISTTProvider(api_key=fake_api_key)
    ws = _RecvFailsSendBlocksWebSocket(
        [{"type": "session.updated", "session": {"type": "transcription"}}],
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "error": {"message": "audio too short"},
        },
    )
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("coval_bench.providers.stt.openai.ws_client.connect", return_value=cm):
        result = await asyncio.wait_for(
            provider.measure_ttft(
                audio_data=audio_pcm_bytes,
                channels=1,
                sample_width=2,
                sample_rate=16000,
                realtime_resolution=0.5,
            ),
            timeout=2.0,
        )

    assert result.error is not None
    assert "audio too short" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_session_ready_timeout_fires(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """The handshake deadline fires when the server never sends session.updated."""
    provider = OpenAISTTProvider(api_key=fake_api_key)
    ws = _SendFailsRecvBlocksWebSocket([])  # recv() blocks; no session.updated ever
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("coval_bench.providers.stt.openai._READY_TIMEOUT_S", 0.05),
        patch("coval_bench.providers.stt.openai.ws_client.connect", return_value=cm),
    ):
        result = await asyncio.wait_for(
            provider.measure_ttft(
                audio_data=audio_pcm_bytes,
                channels=1,
                sample_width=2,
                sample_rate=16000,
                realtime_resolution=0.5,
            ),
            timeout=2.0,
        )

    assert result.error is not None
    assert "session.updated" in result.error


def test_openai_provider_name() -> None:
    provider = OpenAISTTProvider(api_key=SecretStr("test"))
    assert provider.name == "openai-gpt-realtime-whisper"


def test_openai_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid OpenAI STT model"):
        OpenAISTTProvider(api_key=SecretStr("test"), model="whisper-1")


@pytest.mark.asyncio
async def test_openai_rejects_non_positive_realtime_resolution(fake_api_key: SecretStr) -> None:
    provider = OpenAISTTProvider(api_key=fake_api_key)

    with pytest.raises(ValueError, match="realtime_resolution must be > 0"):
        await provider.measure_ttft(
            audio_data=b"\x00\x00",
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0,
        )


@pytest.mark.asyncio
async def test_openai_session_ready_stream_ended(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """Handshake stream ends before session.updated → bounded error (StopAsyncIteration path)."""
    provider = OpenAISTTProvider(api_key=fake_api_key)

    with (
        patch("coval_bench.providers.stt.openai._READY_TIMEOUT_S", 0.05),
        patch(
            "coval_bench.providers.stt.openai.ws_client.connect",
            return_value=_fake_connect(
                [{"type": "session.created", "session": {"id": "sess_slow"}}]
            ),
        ),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "session.updated" in result.error


@pytest.mark.asyncio
async def test_openai_rejects_non_mono_input(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    provider = OpenAISTTProvider(api_key=fake_api_key)

    result = await provider.measure_ttft(
        audio_data=audio_pcm_bytes,
        channels=2,
        sample_width=2,
        sample_rate=16000,
        realtime_resolution=0.5,
    )

    assert result.error is not None
    assert "mono PCM" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_rejects_non_pcm16_input(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    provider = OpenAISTTProvider(api_key=fake_api_key)

    result = await provider.measure_ttft(
        audio_data=audio_pcm_bytes,
        channels=1,
        sample_width=1,
        sample_rate=16000,
        realtime_resolution=0.5,
    )

    assert result.error is not None
    assert "16-bit PCM" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_session_setup_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.openai.ws_client.connect",
        return_value=_fake_connect(
            [
                {
                    "type": "error",
                    "error": {
                        "message": "Turn detection is not supported for this transcription model."
                    },
                }
            ]
        ),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "session setup" in result.error
    assert "Turn detection is not supported" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.openai.ws_client.connect",
        return_value=_fake_connect(load_fixture_events("openai", "events-error")),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "quota exceeded" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_openai_no_transcript(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.openai.ws_client.connect",
        return_value=_fake_connect(load_fixture_events("openai", "events-no-transcript")),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error == "ws_closed_without_completed"
    assert result.ttft_seconds is not None
    assert result.complete_transcript is not None
    assert "hello" in result.complete_transcript


@pytest.mark.asyncio
async def test_openai_item_id_tracking(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = OpenAISTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.openai.ws_client.connect",
        return_value=_fake_connect(load_fixture_events("openai", "events-multi-item")),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    # `.completed` is scoped to its own item_id (no cross-item merge), so item_001's
    # final is only its own deltas; item_002's text stays in the partials.
    assert result.complete_transcript == "hello how"
    assert result.first_token_content is not None
    assert result.first_token_content.startswith("hello")
    assert result.partial_transcripts is not None
    assert any("world are you" in p for p in result.partial_transcripts)
