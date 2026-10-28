# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.krisp (KrispSTTProvider).

The session-token POST is mocked with a fake httpx client and the WebSocket is
replayed with FakeWebSocket — no live network calls are made.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from coval_bench.config import Settings
from coval_bench.providers.stt.krisp import KrispSTTProvider
from tests.providers.fakes import FakeWebSocket, fake_connect

_TOKEN = "session_key_test"  # noqa: S105


def _transcript(words: list[str], *, final: bool) -> dict[str, Any]:
    return {
        "type": "transcript",
        "metadata": {},
        "transcript": {"words": [{"text": w} for w in words], "isFinal": final},
    }


def _http_client(response: httpx.Response) -> Any:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=response)
    return client


def _token_ok() -> httpx.Response:
    return httpx.Response(200, json={"data": {"session_key": _TOKEN, "usage_type": "stt"}})


async def _run(
    settings: Settings,
    audio: bytes,
    events: list[Any],
    token_response: httpx.Response | None = None,
) -> tuple[Any, FakeWebSocket, MagicMock]:
    cm = fake_connect(events)
    connect = MagicMock(return_value=cm)
    with (
        patch(
            "coval_bench.providers.stt.krisp.httpx.AsyncClient",
            return_value=_http_client(token_response or _token_ok()),
        ),
        patch("coval_bench.providers.stt.krisp.ws_client.connect", connect),
    ):
        result = await KrispSTTProvider(settings).measure_ttft(
            audio_data=audio, channels=1, sample_width=2, sample_rate=16000
        )
    return result, cm.ws, connect


@pytest.mark.asyncio
async def test_krisp_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [
        {"type": "ready", "request_id": "r1"},
        _transcript([" hel"], final=False),
        _transcript([" hello"], final=True),
        _transcript([" world", " how"], final=False),
        {"type": "turn", "reason": "model", "confidence": 0.95},
        _transcript([], final=False),
        _transcript([" world", " how", " are", " you"], final=True),
        {"type": "end"},
    ]
    result, ws, connect = await _run(stt_settings, audio_pcm_bytes, events)

    assert result.error is None
    assert result.first_token_content == "hel"  # noqa: S105
    assert result.complete_transcript == "hello world how are you"
    assert result.audio_to_final_seconds is not None

    assert connect.call_args.kwargs["additional_headers"] == {"Authorization": f"api-key {_TOKEN}"}
    hello = json.loads(ws.sent[0])
    assert hello["model"] == "krisp-realtime-stt-agents-v1"
    assert hello["sample_rate"] == 16000
    audio = [m for m in ws.sent if isinstance(m, bytes)]
    assert all(len(chunk) % 2560 == 0 for chunk in audio[:-1])
    assert json.loads(ws.sent[-1]) == {"type": "end_of_audio"}


@pytest.mark.asyncio
async def test_krisp_mid_stream_error_fails(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [
        {"type": "ready"},
        {
            "type": "error",
            "request_id": "r2",
            "error": {"code": "balance_exhausted", "status_code": 403, "message": "No credit"},
        },
    ]
    result, _, _ = await _run(stt_settings, audio_pcm_bytes, events)

    assert result.error is not None
    assert "No credit" in result.error
    assert "r2" in result.error


@pytest.mark.asyncio
async def test_krisp_token_failure_fails(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    result, _, connect = await _run(
        stt_settings, audio_pcm_bytes, [], token_response=httpx.Response(401, text="bad key")
    )

    assert result.error is not None
    assert "401" in result.error
    connect.assert_not_called()
