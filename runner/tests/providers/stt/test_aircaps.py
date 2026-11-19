# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.aircaps (AircapsSTTProvider)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt.aircaps import AircapsSTTProvider, _AircapsTranscription
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import make_settings

_WS_URL = "wss://rt1.aircaps.ai/v1/realtime"
_AUTH = {"Authorization": "Bearer test-api-key-00000000000000000000"}


def _http_client(response: httpx.Response) -> MagicMock:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=response)
    return client


def _transcript(text: str, final_chars: int, *, final: bool) -> dict[str, Any]:
    return {"type": "transcript", "text": text, "final_chars": final_chars, "is_final": final}


async def _run(
    settings: Settings, audio: bytes, events: list[Any], sent: list[Any] | None = None
) -> tuple[Any, MagicMock, MagicMock]:
    lookup = httpx.Response(200, json={"websocket_url": _WS_URL, "model": "aircaps-live-1"})
    http = _http_client(lookup)
    connect = MagicMock(return_value=fake_connect(events, sent))
    with (
        patch("coval_bench.providers.stt.aircaps.httpx.AsyncClient", return_value=http),
        patch("coval_bench.providers.stt.aircaps.ws_client.connect", connect),
    ):
        result = await AircapsSTTProvider(settings).measure_ttft(audio, 1, 2, 16000, 0.5)
    return result, http, connect


@pytest.mark.asyncio
async def test_aircaps_success_and_wire_shape(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [
        {"type": "status", "message": "warming"},
        {"type": "ready"},
        _transcript("Hello", 0, final=False),
        _transcript("Hello world, how", 5, final=False),
        _transcript("Hello world, how are you?", 25, final=True),
        _transcript("Hello world, how are you?", 25, final=True),
        {"type": "finished"},
    ]
    sent: list[Any] = []
    result, http, connect = await _run(stt_settings, audio_pcm_bytes, events, sent)

    assert result.error is None
    assert result.provider == "aircaps"
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is not None
    assert result.complete_transcript == "Hello world, how are you?"

    http.get.assert_awaited_once_with(
        "https://api.aircaps.com/v1/realtime/endpoint",
        params={"model": "aircaps-live-1"},
        headers=_AUTH,
    )
    assert connect.call_args.args[0] == _WS_URL
    assert connect.call_args.kwargs["additional_headers"] == _AUTH
    control = [json.loads(m) for m in sent if isinstance(m, str)]
    assert control == [
        {
            "type": "start",
            "model": "aircaps-live-1",
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
        },
        {"type": "finalize"},
        {"type": "finish"},
    ]
    assert b"".join(m for m in sent if isinstance(m, bytes)) == audio_pcm_bytes


@pytest.mark.asyncio
async def test_aircaps_refused_session_fails_the_clip(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    error = {"type": "error", "code": "quota_unavailable", "message": "too many sessions"}
    result, _, _ = await _run(stt_settings, audio_pcm_bytes, [error])

    assert result.error is not None
    assert "quota_unavailable: too many sessions" in result.error
    assert result.complete_transcript is None


def test_aircaps_final_after_finalize_keeps_first_timing(stt_settings: Settings) -> None:
    provider = AircapsSTTProvider(stt_settings)
    tx = _AircapsTranscription(TranscriptionResult(provider="aircaps", audio_start_time=10.0))
    final = _transcript("Hello world", 11, final=True)

    provider._on_message(final, 10.5, tx)
    tx.finalized = True
    provider._on_message(final, 10.9, tx)

    assert tx.result.audio_to_final_seconds == pytest.approx(0.5)
    assert tx.finals == ["Hello world"]


def test_aircaps_construction_guards() -> None:
    with pytest.raises(ValueError, match="aircaps_api_key is required"):
        AircapsSTTProvider(make_settings(aircaps_api_key=None))
