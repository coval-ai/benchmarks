# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.stepfun (StepfunSTTProvider).

All tests use FakeWebSocket. The fixtures mirror the StepFun realtime wire
protocol: deltas carry the cumulative ``text`` plus a correctable ``stash``
tail, and ``conversation.item.input_audio_transcription.completed`` finalizes
the committed buffer.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.stepfun import StepfunSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events

_CONNECT = "coval_bench.providers.stt.stepfun.ws_client.connect"
_MODEL = "stt-model-under-test"
_UPDATED = {"type": "session.updated", "session": {}}


def _fake_connect(events: list[Any], *, captured: dict[str, Any] | None = None) -> Any:
    ws = FakeWebSocket(events)
    if captured is not None:
        captured["ws"] = ws
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


async def _measure(provider: StepfunSTTProvider, audio: bytes, **overrides: Any) -> Any:
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
async def test_stepfun_success_and_wire_shape(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = StepfunSTTProvider(api_key=fake_api_key, model=_MODEL)
    captured: dict[str, Any] = {}

    def connect_side_effect(
        url: str, additional_headers: dict[str, str] | None = None, **_: Any
    ) -> Any:
        captured.update(url=url, headers=dict(additional_headers or {}))
        return _fake_connect(load_fixture_events("stepfun"), captured=captured)

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

    assert captured["url"] == "wss://api.stepfun.ai/v1/realtime/asr/stream"
    assert captured["headers"] == {"Authorization": f"Bearer {fake_api_key.get_secret_value()}"}

    sent = [json.loads(m) for m in captured["ws"]._sent if isinstance(m, str)]
    assert all(m["event_id"] for m in sent)
    assert sent[0]["type"] == "session.update"
    assert sent[0]["session"]["audio"]["input"] == {
        "format": {"type": "pcm", "codec": "pcm_s16le", "rate": 16000, "bits": 16, "channel": 1},
        "transcription": {"model": _MODEL, "language": "en"},
    }
    assert sent[-1]["type"] == "input_audio_buffer.commit"
    appends = sent[1:-1]
    assert appends and all(m["type"] == "input_audio_buffer.append" for m in appends)
    assert len(base64.b64decode(appends[0]["audio"])) == 16000  # 0.5 s x 16 kHz x 2 bytes
    assert b"".join(base64.b64decode(m["audio"]) for m in appends) == audio_pcm_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (
            load_fixture_events("stepfun", "events-error"),
            "Invalid parameter value (code=invalid_value)",
        ),
        ([{"type": "error", "error": {"message": "Invalid API key"}}], "session setup"),
    ],
    ids=["error-event", "setup-error"],
)
async def test_stepfun_failures_surface_as_result_error(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes, events: list[Any], expected: str
) -> None:
    provider = StepfunSTTProvider(api_key=fake_api_key, model=_MODEL)

    with patch(_CONNECT, return_value=_fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert expected in result.error
    assert result.complete_transcript is None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_stepfun_session_ready_timeout(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = StepfunSTTProvider(api_key=fake_api_key, model=_MODEL)

    with (
        patch("coval_bench.providers.stt.stepfun._READY_TIMEOUT_S", 0.05),
        patch(_CONNECT, return_value=_fake_connect([])),
    ):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error is not None
    assert "session.updated" in result.error


@pytest.mark.asyncio
async def test_stepfun_closed_without_completed_keeps_longest_partial(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    delta = "conversation.item.input_audio_transcription.delta"
    events = [
        _UPDATED,
        {"type": delta, "item_id": "item_001", "text": "", "stash": "hello"},
        {"type": delta, "item_id": "item_001", "text": "hello ", "stash": "world"},
    ]
    provider = StepfunSTTProvider(api_key=fake_api_key, model=_MODEL)

    with patch(_CONNECT, return_value=_fake_connect(events)):
        result = await _measure(provider, audio_pcm_bytes)

    assert result.error == "ws_closed_without_completed"
    assert result.complete_transcript == "hello world"


@pytest.mark.asyncio
async def test_stepfun_rejects_wrong_audio_format(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = StepfunSTTProvider(api_key=fake_api_key, model=_MODEL)

    wrong_rate = await _measure(provider, audio_pcm_bytes, sample_rate=8000)
    assert wrong_rate.error is not None
    assert "16 kHz" in wrong_rate.error

    stereo = await _measure(provider, audio_pcm_bytes, channels=2)
    assert stereo.error is not None
    assert "mono 16-bit" in stereo.error


def test_stepfun_construction_guards() -> None:
    provider = StepfunSTTProvider(api_key=SecretStr("k"), model=_MODEL)
    assert (provider.name, provider.model) == ("stepfun", _MODEL)
    with pytest.raises(ValueError, match="stepfun_api_key is required"):
        StepfunSTTProvider(api_key=None, model=_MODEL)
