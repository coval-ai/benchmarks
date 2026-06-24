# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.gladia (GladiaSTTProvider).

The session-init POST is mocked with a fake httpx client; the WebSocket is
replayed with FakeWebSocket — no live network calls are made. Each final
``transcript`` message is a complete utterance, so the transcript is the
in-order join of every final.
"""

from __future__ import annotations

from asyncio import sleep as _real_sleep
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.gladia import GladiaSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events

_WS_URL = "wss://api.gladia.io/v2/live/test-session"


def make_provider() -> GladiaSTTProvider:
    return GladiaSTTProvider(api_key=SecretStr("test-key-gladia"))


def _fake_ws_connect(events: list[Any]) -> Any:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _fake_http_client(response: httpx.Response) -> Any:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=response)
    return client


def _init_ok() -> httpx.Response:
    return httpx.Response(200, json={"id": "test-session", "url": _WS_URL})


@pytest.fixture(autouse=True)
def _no_realtime_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the real-time pacing sleep so streaming tests run instantly."""

    async def _noop(_seconds: float) -> None:
        return None

    monkeypatch.setattr("coval_bench.providers.stt.gladia.asyncio.sleep", _noop)


# ---------------------------------------------------------------------------
# Happy path — finals joined in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gladia_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("gladia", "events-success")
    provider = GladiaSTTProvider(api_key=fake_api_key)

    with (
        patch(
            "coval_bench.providers.stt.gladia.httpx.AsyncClient",
            return_value=_fake_http_client(_init_ok()),
        ),
        patch(
            "coval_bench.providers.stt.gladia.ws_client.connect",
            return_value=_fake_ws_connect(events),
        ),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
        )

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content == "hello"  # noqa: S105
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    assert result.audio_to_final_seconds is not None
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_gladia_partials_only_falls_back_to_longest(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """With no final, the longest partial is salvaged as the transcript."""
    events: list[Any] = [
        {"type": "transcript", "data": {"is_final": False, "utterance": {"text": "hello"}}},
        {"type": "transcript", "data": {"is_final": False, "utterance": {"text": "hello world"}}},
    ]
    provider = GladiaSTTProvider(api_key=fake_api_key)

    with (
        patch(
            "coval_bench.providers.stt.gladia.httpx.AsyncClient",
            return_value=_fake_http_client(_init_ok()),
        ),
        patch(
            "coval_bench.providers.stt.gladia.ws_client.connect",
            return_value=_fake_ws_connect(events),
        ),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
        )

    assert result.error is None
    assert result.ttft_seconds is not None  # first partial still marks first token
    assert result.complete_transcript == "hello world"
    assert result.audio_to_final_seconds is None


# ---------------------------------------------------------------------------
# Provider name and model
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    assert make_provider().name == "gladia-solaria-1"


def test_provider_model() -> None:
    assert make_provider().model == "solaria-1"


# ---------------------------------------------------------------------------
# Construction errors
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Gladia STT model"):
        GladiaSTTProvider(api_key=SecretStr("k"), model="solaria-3")


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="gladia_api_key is required"):
        GladiaSTTProvider(api_key=None)


# ---------------------------------------------------------------------------
# Failure path — session init rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gladia_init_failure(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = GladiaSTTProvider(api_key=fake_api_key)
    response = httpx.Response(401, json={"message": "Invalid API key"})

    with patch(
        "coval_bench.providers.stt.gladia.httpx.AsyncClient",
        return_value=_fake_http_client(response),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
        )

    assert result.error is not None
    assert "session init failed" in result.error
    assert "401" in result.error
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


# ---------------------------------------------------------------------------
# Failure path — error event over the socket
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gladia_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("gladia", "events-error")
    provider = GladiaSTTProvider(api_key=fake_api_key)

    with (
        patch(
            "coval_bench.providers.stt.gladia.httpx.AsyncClient",
            return_value=_fake_http_client(_init_ok()),
        ),
        patch(
            "coval_bench.providers.stt.gladia.ws_client.connect",
            return_value=_fake_ws_connect(events),
        ),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
        )

    assert result.error is not None
    assert "Invalid or expired API key" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_gladia_cancels_sender_when_receiver_errors(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receiver error cancels the still-streaming sender instead of draining it."""
    # Real yielding sleep (overriding the autouse no-op) so the sender is mid-stream.
    monkeypatch.setattr("coval_bench.providers.stt.gladia.asyncio.sleep", lambda _s: _real_sleep(0))
    sent: list[Any] = []
    ws = FakeWebSocket(load_fixture_events("gladia", "events-error"), on_send=sent.append)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    provider = GladiaSTTProvider(api_key=fake_api_key)

    with (
        patch(
            "coval_bench.providers.stt.gladia.httpx.AsyncClient",
            return_value=_fake_http_client(_init_ok()),
        ),
        patch("coval_bench.providers.stt.gladia.ws_client.connect", return_value=cm),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes, channels=1, sample_width=2, sample_rate=16000
        )

    assert result.error is not None
    assert "Invalid or expired API key" in result.error
    assert sent
    # stop_recording is the sender's only text frame; its absence == cancelled mid-stream.
    assert not any(isinstance(frame, str) for frame in sent)


# ---------------------------------------------------------------------------
# Failure path — empty stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gladia_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = GladiaSTTProvider(api_key=fake_api_key)

    with (
        patch(
            "coval_bench.providers.stt.gladia.httpx.AsyncClient",
            return_value=_fake_http_client(_init_ok()),
        ),
        patch(
            "coval_bench.providers.stt.gladia.ws_client.connect",
            return_value=_fake_ws_connect([]),
        ),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
        )

    assert result.complete_transcript is None
    assert result.ttft_seconds is None
