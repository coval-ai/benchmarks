# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.nari (NariSTTProvider).

All tests use FakeWebSocket — no live network calls are made. The fixtures
mirror the Nari realtime wire protocol: ``transcript.partial`` replaces the
running hypothesis, ``transcript.completed`` finalizes one ``item_id``, and a
long recording is split server-side into several items whose earlier
completions carry ``commit_reason: "max_duration"``.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.nari import NariSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events

_CONNECT = "coval_bench.providers.stt.nari.ws_client.connect"
_CONFIGURED = {"type": "session.configured", "session": {"id": "s"}}


def _fake_connect(events: list[Any], *, captured: dict[str, Any] | None = None) -> Any:
    ws = FakeWebSocket(events)
    if captured is not None:
        captured["ws"] = ws
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


async def _measure(provider: NariSTTProvider, audio: bytes, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "audio_data": audio,
        "channels": 1,
        "sample_width": 2,
        "sample_rate": 16000,
        "realtime_resolution": 0.5,
    }
    kwargs.update(overrides)
    return await provider.measure_ttft(**kwargs)


@pytest.mark.asyncio
async def test_nari_success_and_wire_shape(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = NariSTTProvider(api_key=fake_api_key)
    captured: dict[str, Any] = {}

    def connect_side_effect(
        url: str, additional_headers: dict[str, str] | None = None, **_: Any
    ) -> Any:
        captured.update(url=url, headers=dict(additional_headers or {}))
        return _fake_connect(load_fixture_events("nari"), captured=captured)

    with patch(_CONNECT, side_effect=connect_side_effect):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is not None
    assert 0 <= result.ttft_seconds <= result.audio_to_final_seconds
    assert result.first_token_content == "hello"  # noqa: S105
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    assert "hello world how" in result.partial_transcripts

    parsed = urlparse(captured["url"])
    assert (parsed.scheme, parsed.netloc, parsed.path) == (
        "wss",
        "api.narilabs.com",
        "/v1/realtime",
    )
    assert parse_qs(parsed.query) == {"intent": ["transcription"]}
    assert captured["headers"] == {"Authorization": f"Bearer {fake_api_key.get_secret_value()}"}

    sent = [json.loads(m) for m in captured["ws"]._sent if isinstance(m, str)]
    assert sent[0] == {
        "type": "session.configure",
        "session": {"model": "qwen3-asr-fast", "language": "en", "turn_detection": None},
    }
    assert sent[-1] == {"type": "input_audio_buffer.commit"}
    appends = sent[1:-1]
    assert appends and all(m["type"] == "input_audio_buffer.append" for m in appends)
    assert len(base64.b64decode(appends[0]["audio"])) == 16000  # 0.5 s x 16 kHz x 2 bytes
    assert b"".join(base64.b64decode(m["audio"]) for m in appends) == audio_pcm_bytes


@pytest.mark.asyncio
async def test_nari_joins_auto_split_items_in_order(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = NariSTTProvider(api_key=fake_api_key)

    with patch(_CONNECT, return_value=_fake_connect(load_fixture_events("nari", "events-split"))):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is None
    assert result.complete_transcript == "hello world how are you"
    assert result.first_token_content == "hello"  # noqa: S105
    assert result.audio_to_final_seconds is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (load_fixture_events("nari", "events-error"), "Insufficient credits (requestId=req-abc)"),
        ([{"type": "error", "error": {"message": "Invalid API key"}}], "session setup"),
        ([_CONFIGURED, {"type": "input_audio_buffer.commit_empty"}], "no pending audio"),
    ],
    ids=["error-event", "setup-error", "commit-empty"],
)
async def test_nari_failures_surface_as_result_error(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes, events: list[Any], expected: str
) -> None:
    provider = NariSTTProvider(api_key=fake_api_key)

    with patch(_CONNECT, return_value=_fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert expected in result.error
    assert result.complete_transcript is None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_nari_session_ready_timeout(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    provider = NariSTTProvider(api_key=fake_api_key)

    with (
        patch("coval_bench.providers.stt.nari._READY_TIMEOUT_S", 0.05),
        patch(_CONNECT, return_value=_fake_connect([])),
    ):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert "session.configured" in result.error


@pytest.mark.asyncio
async def test_nari_closed_without_completed_keeps_longest_partial(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events = [
        _CONFIGURED,
        {"type": "transcript.partial", "item_id": "item_001", "transcript": "hello"},
        {"type": "transcript.partial", "item_id": "item_001", "transcript": "hello world"},
    ]
    provider = NariSTTProvider(api_key=fake_api_key)

    with patch(_CONNECT, return_value=_fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error == "ws_closed_without_completed"
    assert result.complete_transcript == "hello world"


@pytest.mark.asyncio
async def test_nari_rejects_wrong_audio_format(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = NariSTTProvider(api_key=fake_api_key)

    wrong_rate = await _measure(provider, audio_pcm_bytes, sample_rate=8000)
    assert wrong_rate.error is not None
    assert "16 kHz" in wrong_rate.error

    stereo = await _measure(provider, audio_pcm_bytes, channels=2)
    assert stereo.error is not None
    assert "mono 16-bit" in stereo.error


def test_nari_construction_guards() -> None:
    provider = NariSTTProvider(api_key=SecretStr("k"))
    assert (provider.name, provider.model) == ("nari", "qwen3-asr-fast")
    with pytest.raises(ValueError, match="Invalid Nari STT model"):
        NariSTTProvider(api_key=SecretStr("k"), model="qwen3-asr-fast:free")
    with pytest.raises(ValueError, match="nari_api_key is required"):
        NariSTTProvider(api_key=None)
