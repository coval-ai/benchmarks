# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.cartesia (CartesiaSTTProvider).

All tests use FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.cartesia import CartesiaSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events


def make_provider() -> CartesiaSTTProvider:
    return CartesiaSTTProvider(api_key=SecretStr("test-key-cartesia"))


def _fake_connect(events: list[Any], sent: list[Any] | None = None) -> Any:
    ws = FakeWebSocket(events, on_send=(sent.append if sent is not None else None))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cartesia_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("cartesia", "events-success")
    provider = CartesiaSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.cartesia.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
            audio_duration=3.0,
        )

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert "hello" in result.first_token_content.lower()
    assert result.complete_transcript == "hello world how are you"


# ---------------------------------------------------------------------------
# Provider name and model
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    assert make_provider().name == "cartesia-ink-2"


def test_provider_model() -> None:
    assert make_provider().model == "ink-2"


# ---------------------------------------------------------------------------
# Invalid model / mono guard
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Cartesia STT model"):
        CartesiaSTTProvider(api_key=SecretStr("k"), model="ink-whisper")


@pytest.mark.asyncio
async def test_stereo_rejected(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = CartesiaSTTProvider(api_key=fake_api_key)
    with pytest.raises(ValueError, match="mono"):
        await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=2,
            sample_width=2,
            sample_rate=16000,
        )


@pytest.mark.asyncio
async def test_non_16bit_rejected(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = CartesiaSTTProvider(api_key=fake_api_key)
    with pytest.raises(ValueError, match="16-bit PCM"):
        await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=4,
            sample_rate=16000,
        )


# ---------------------------------------------------------------------------
# Control commands: finalize then close are sent after the audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_and_close_sent(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("cartesia", "events-success")
    sent: list[Any] = []
    provider = CartesiaSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.cartesia.ws_client.connect",
        return_value=_fake_connect(events, sent=sent),
    ):
        await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    text_cmds = [m for m in sent if isinstance(m, str)]
    assert text_cmds == ["finalize", "close"]


# ---------------------------------------------------------------------------
# Connection URL, auth header, and pinned API version (regression guard for the
# X-API-Key / wrong-version bug the hermetic suite originally missed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_url_and_auth(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("cartesia", "events-success")
    connect_mock = MagicMock(return_value=_fake_connect(events))
    provider = CartesiaSTTProvider(api_key=fake_api_key)

    with patch("coval_bench.providers.stt.cartesia.ws_client.connect", connect_mock):
        await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    url = connect_mock.call_args.args[0]
    assert url.startswith("wss://api.cartesia.ai/stt/websocket?")
    for param in ("model=ink-2", "encoding=pcm_s16le", "sample_rate=16000", "language=en"):
        assert param in url
    assert "cartesia_version" not in url  # version is a header, not a query param

    headers = connect_mock.call_args.kwargs["additional_headers"]
    assert headers["Authorization"] == f"Bearer {fake_api_key.get_secret_value()}"
    assert headers["cartesia-version"] == "2025-11-04"


# ---------------------------------------------------------------------------
# Field-name guard: transcript text lives in "text", not "transcript"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reads_text_field_not_transcript(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [
        {"type": "transcript", "is_final": True, "transcript": "wrong field", "request_id": "r"},
        {"type": "done", "request_id": "r"},
    ]
    provider = CartesiaSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.cartesia.ws_client.connect",
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
    assert result.ttft_seconds is None


# ---------------------------------------------------------------------------
# Empty stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cartesia_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = CartesiaSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.cartesia.ws_client.connect",
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
# WS-close gate: a stream that ends WITHOUT a "done" marker still terminates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_done_marker_terminates(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [
        {"type": "transcript", "is_final": True, "text": "hello world", "request_id": "r"},
    ]
    provider = CartesiaSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.cartesia.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.complete_transcript == "hello world"
    assert result.audio_to_final_seconds is not None


# ---------------------------------------------------------------------------
# audio_to_final: set on a final, None when only partials arrive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_to_final_none_on_partial_only(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [
        {"type": "transcript", "is_final": False, "text": "hello", "request_id": "r"},
        {"type": "done", "request_id": "r"},
    ]
    provider = CartesiaSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.cartesia.ws_client.connect",
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
    assert result.ttft_seconds is not None
    # No is_final arrived: TTFT is still measured, but the transcript stays
    # incomplete so the orchestrator does not score WER from a fragment.
    assert result.complete_transcript is None


# ---------------------------------------------------------------------------
# Error event populates result.error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_event_sets_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [
        {"type": "error", "status_code": 400, "title": "Bad", "message": "boom"},
    ]
    provider = CartesiaSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.cartesia.ws_client.connect",
        return_value=_fake_connect(events),
    ):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error == "boom"
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_malformed_frame_sets_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    """A non-JSON frame must surface as result.error, not a silent clean run."""
    provider = CartesiaSTTProvider(api_key=fake_api_key)

    with patch(
        "coval_bench.providers.stt.cartesia.ws_client.connect",
        return_value=_fake_connect(["not-json{"]),
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
