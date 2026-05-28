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
async def test_xai_cumulative_is_final_without_done_text(
    fake_api_key: SecretStr,
    audio_pcm_bytes: bytes,
) -> None:
    """is_final partials are cumulative; empty transcript.done uses last is_final only."""
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
    assert query.get("language") == ["en"]
    auth = captured["additional_headers"].get("Authorization")
    assert auth == f"Bearer {fake_api_key.get_secret_value()}"

    sent_binary = [msg for msg in captured["ws"]._sent if isinstance(msg, bytes)]
    sent_json = [json.loads(msg) for msg in captured["ws"]._sent if isinstance(msg, str)]
    assert sent_binary, "expected raw audio frames to be sent"
    assert sent_json, "expected at least one JSON control message to be sent"
    assert sent_json[-1]["type"] == "audio.done"


def test_xai_provider_name() -> None:
    provider = XaiSTTProvider(api_key=SecretStr("test"))
    assert provider.name == "xai-grok-stt"


def test_xai_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid xAI STT model"):
        XaiSTTProvider(api_key=SecretStr("k"), model="bad-model")


@pytest.mark.asyncio
async def test_xai_rejects_non_pcm16(fake_api_key: SecretStr) -> None:
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with pytest.raises(ValueError, match="16-bit PCM"):
        await provider.measure_ttft(
            audio_data=b"\x00" * 16,
            channels=1,
            sample_width=1,
            sample_rate=16000,
        )


@pytest.mark.asyncio
async def test_xai_rejects_non_positive_realtime_resolution(fake_api_key: SecretStr) -> None:
    provider = XaiSTTProvider(api_key=fake_api_key, model="grok-stt")

    with pytest.raises(ValueError, match="realtime_resolution must be > 0"):
        await provider.measure_ttft(
            audio_data=b"\x00" * 16,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0,
        )


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
