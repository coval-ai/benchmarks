# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.reson8 (Reson8STTProvider).

All tests use FakeWebSocket — no live network calls. Fixtures mirror the Reson8
realtime protocol: ``transcript`` frames carrying ``is_final``, and a
``flush_confirmation`` acknowledging the client's ``flush_request``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.reson8 import Reson8STTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def make_provider() -> Reson8STTProvider:
    return Reson8STTProvider(api_key=SecretStr("test-key-reson8"))


def _fake_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class FlushGatedWebSocket:
    """Fake WS that withholds the tail final and the ack until a flush is sent.

    ``FakeWebSocket`` preloads every response and ends the stream once they are
    consumed, so the ack lands whether or not the client asked for it. This one
    keeps the post-flush frames queued until a ``flush_request`` actually
    arrives, which is what exercises the ordering that protects the tail.
    """

    def __init__(
        self,
        interims: list[dict[str, Any] | str],
        post_flush: list[dict[str, Any] | str],
    ) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        for event in interims:
            self._queue.put_nowait(self._encode(event))
        self._post_flush = post_flush
        self.sent: list[Any] = []
        self.flush_seen = False

    @staticmethod
    def _encode(event: dict[str, Any] | str) -> str:
        """Dicts are serialised; raw strings go on the wire verbatim."""
        return json.dumps(event) if isinstance(event, dict) else event

    async def send(self, msg: bytes | str) -> None:
        self.sent.append(msg)
        if isinstance(msg, str) and json.loads(msg).get("type") == "flush_request":
            self.flush_seen = True
            for event in self._post_flush:
                self._queue.put_nowait(self._encode(event))
            self._queue.put_nowait(None)

    async def close(self) -> None:
        self._queue.put_nowait(None)

    def __aiter__(self) -> FlushGatedWebSocket:
        return self

    async def __anext__(self) -> str:
        # Bounded so a client that never flushes fails the test instead of hanging.
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=2.0)
        except TimeoutError:
            raise StopAsyncIteration from None
        if item is None:
            raise StopAsyncIteration
        return item


def _gated_connect(ws: FlushGatedWebSocket) -> Any:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


async def _run(provider: Reson8STTProvider, ws_cm: Any, audio: bytes) -> Any:
    with patch("coval_bench.providers.stt.reson8.ws_client.connect", return_value=ws_cm):
        return await provider.measure_ttft(
            audio_data=audio,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reson8_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("reson8", "events-success")
    provider = Reson8STTProvider(api_key=fake_api_key)

    result = await _run(provider, _fake_connect(events), audio_pcm_bytes)

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content == "hello"  # noqa: S105 - transcript fixture text
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    assert result.audio_to_final_seconds is not None
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_reson8_ttft_fires_on_interim(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """TTFT is time-to-first-token: an interim sets it, ahead of any final."""
    events: list[Any] = [
        {"type": "transcript", "text": "hello", "is_final": False},
        {"type": "transcript", "text": "hello world", "is_final": True},
        {"type": "flush_confirmation", "id": "eos"},
    ]
    result = await _run(
        Reson8STTProvider(api_key=fake_api_key), _fake_connect(events), audio_pcm_bytes
    )

    assert result.error is None
    assert result.first_token_content == "hello"  # noqa: S105 - transcript fixture text
    assert result.partial_transcripts == ["hello"]


@pytest.mark.asyncio
async def test_reson8_excludes_interims_from_transcript(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [
        {"type": "transcript", "text": "wrong guess entirely", "is_final": False},
        {"type": "transcript", "text": "hello world", "is_final": True},
        {"type": "flush_confirmation", "id": "eos"},
    ]
    result = await _run(
        Reson8STTProvider(api_key=fake_api_key), _fake_connect(events), audio_pcm_bytes
    )

    assert result.complete_transcript == "hello world"
    assert result.word_count == 2


@pytest.mark.asyncio
async def test_reson8_joins_segment_finals(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [
        {"type": "transcript", "text": "hello world", "is_final": True},
        {"type": "transcript", "text": "how are you", "is_final": True},
        {"type": "flush_confirmation", "id": "eos"},
    ]
    result = await _run(
        Reson8STTProvider(api_key=fake_api_key), _fake_connect(events), audio_pcm_bytes
    )

    assert result.complete_transcript == "hello world how are you"


@pytest.mark.asyncio
async def test_reson8_treats_missing_is_final_as_final(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """The flag is absent when interims are off; those transcripts are finals."""
    events: list[Any] = [
        {"type": "transcript", "text": "hello world"},
        {"type": "flush_confirmation", "id": "eos"},
    ]
    result = await _run(
        Reson8STTProvider(api_key=fake_api_key), _fake_connect(events), audio_pcm_bytes
    )

    assert result.error is None
    assert result.complete_transcript == "hello world"
    assert result.partial_transcripts == []


@pytest.mark.asyncio
async def test_reson8_ignores_unknown_frames(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [
        {"type": "session_info", "region": "eu"},
        {"type": "transcript", "text": "hello world", "is_final": True},
        {"type": "flush_confirmation", "id": "eos"},
    ]
    result = await _run(
        Reson8STTProvider(api_key=fake_api_key), _fake_connect(events), audio_pcm_bytes
    )

    assert result.error is None
    assert result.complete_transcript == "hello world"


# ---------------------------------------------------------------------------
# Flush handshake
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reson8_flushes_after_audio(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """Audio goes out as binary frames, then a flush_request forces the tail final."""
    ws = FakeWebSocket([{"type": "flush_confirmation", "id": "eos"}])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    await _run(Reson8STTProvider(api_key=fake_api_key), cm, audio_pcm_bytes)

    assert all(isinstance(frame, bytes) for frame in ws._sent[:-1])
    assert ws._sent[-1] == '{"type": "flush_request", "id": "eos"}'


def test_reson8_url_config(fake_api_key: SecretStr) -> None:
    url = Reson8STTProvider(api_key=fake_api_key)._build_websocket_url(16000)

    assert url.startswith("wss://api.reson8.dev/v1/speech-to-text/realtime?")
    assert "encoding=pcm_s16le" in url
    assert "sample_rate=16000" in url
    assert "channels=1" in url
    assert "language=en" in url
    assert "include_interim=true" in url


def test_reson8_url_omits_credentials(fake_api_key: SecretStr) -> None:
    url = Reson8STTProvider(api_key=fake_api_key)._build_websocket_url(16000)

    assert fake_api_key.get_secret_value() not in url
    assert "ApiKey" not in url


# ---------------------------------------------------------------------------
# Provider identity and construction
# ---------------------------------------------------------------------------


def test_provider_name_and_model() -> None:
    provider = make_provider()
    assert provider.name == "reson8"
    assert provider.model == "realtime"


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Reson8 STT model"):
        Reson8STTProvider(api_key=SecretStr("k"), model="turns")


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="reson8_api_key is required"):
        Reson8STTProvider(api_key=None)


def test_blank_api_key_raises() -> None:
    with pytest.raises(ValueError, match="reson8_api_key is required"):
        Reson8STTProvider(api_key=SecretStr("   "))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reson8_rejects_non_mono_or_non_16bit(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = Reson8STTProvider(api_key=fake_api_key)
    result = await provider.measure_ttft(
        audio_data=audio_pcm_bytes,
        channels=2,
        sample_width=2,
        sample_rate=16000,
    )

    assert result.error is not None
    assert "mono 16-bit" in result.error
    assert result.ttft_seconds is None


# ---------------------------------------------------------------------------
# Error frames — every documented shape must surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        ({"type": "error", "message": "unsupported audio format"}, "unsupported audio format"),
        (
            {"code": "UNAUTHORIZED", "message": "Invalid or expired credentials"},
            "Invalid or expired credentials",
        ),
        ({"error": {"code": "INTERNAL_ERROR"}}, "INTERNAL_ERROR"),
        ({"error": "boom"}, "boom"),
        ({"details": {"code": "INVALID_REQUEST", "message": "bad sample_rate"}}, "bad sample_rate"),
    ],
)
@pytest.mark.asyncio
async def test_reson8_error_frames_surface(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
    frame: dict[str, Any],
    expected: str,
) -> None:
    result = await _run(
        Reson8STTProvider(api_key=fake_api_key), _fake_connect([frame]), audio_pcm_bytes
    )

    assert result.error == expected
    assert result.complete_transcript is None


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reson8_close_without_final_fails(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A clean close carrying only interims is a failure, not an empty success."""
    events: list[Any] = [{"type": "transcript", "text": "hello", "is_final": False}]
    result = await _run(
        Reson8STTProvider(api_key=fake_api_key), _fake_connect(events), audio_pcm_bytes
    )

    assert result.error == "ws_closed_without_final_transcript"


@pytest.mark.asyncio
async def test_reson8_unconfirmed_flush_keeps_the_row(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A final arrived, so the measurement stands even without the flush ack."""
    events: list[Any] = [{"type": "transcript", "text": "hello world", "is_final": True}]
    result = await _run(
        Reson8STTProvider(api_key=fake_api_key), _fake_connect(events), audio_pcm_bytes
    )

    assert result.error is None
    assert result.complete_transcript == "hello world"
    assert result.audio_to_final_seconds is not None


@pytest.mark.asyncio
async def test_reson8_empty_stream_fails(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    result = await _run(Reson8STTProvider(api_key=fake_api_key), _fake_connect([]), audio_pcm_bytes)

    assert result.error == "ws_closed_without_final_transcript"
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_reson8_connection_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = Reson8STTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.reson8.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_reson8_surfaces_send_failure(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A send failure inside the streaming task is surfaced, not swallowed by gather."""

    def _raise_on_audio(msg: object) -> None:
        if isinstance(msg, (bytes, bytearray)):
            raise RuntimeError("send boom")

    ws = FakeWebSocket([], on_send=_raise_on_audio)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    result = await _run(Reson8STTProvider(api_key=fake_api_key), cm, audio_pcm_bytes)

    assert result.error is not None
    assert "send boom" in result.error
    assert result.complete_transcript is None


# ---------------------------------------------------------------------------
# Flush ordering — the tail final only exists because the client asked for it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reson8_tail_final_requires_the_flush(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """The last segment arrives only after flush_request, and must be captured."""
    ws = FlushGatedWebSocket(
        interims=[{"type": "transcript", "text": "hello", "is_final": False}],
        post_flush=[
            {"type": "transcript", "text": "hello world", "is_final": True},
            {"type": "flush_confirmation", "id": "eos"},
        ],
    )

    result = await _run(
        Reson8STTProvider(api_key=fake_api_key), _gated_connect(ws), audio_pcm_bytes
    )

    assert ws.flush_seen
    assert result.error is None
    assert result.complete_transcript == "hello world"
    assert result.partial_transcripts == ["hello"]
    assert result.audio_to_final_seconds is not None


@pytest.mark.asyncio
async def test_reson8_receive_failure_after_a_segment_final_fails_the_row(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """A mid-stream failure after segment one must not publish a truncated transcript."""
    ws = FlushGatedWebSocket(
        interims=[
            {"type": "transcript", "text": "first segment", "is_final": True},
            "{not json",
        ],
        post_flush=[],
    )

    result = await _run(
        Reson8STTProvider(api_key=fake_api_key), _gated_connect(ws), audio_pcm_bytes
    )

    assert result.error is not None
    assert "ws_closed_without_final_transcript" not in result.error
