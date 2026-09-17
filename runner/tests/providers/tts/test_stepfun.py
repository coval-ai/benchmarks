# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the StepFun WebSocket TTS provider."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.stepfun import StepfunTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_CONNECT = "coval_bench.providers.tts.stepfun.ws_client.connect"
_MODEL = "tts-model-under-test"
_VOICE = "voice-under-test"
_SESSION = "01956e7388477cfcbdc3aaabf364bc70"


def _event(event_type: str, **data: object) -> str:
    return json.dumps(
        {"event_id": "evt", "type": event_type, "data": {"session_id": _SESSION, **data}}
    )


def _session_events(pcm_chunks: list[bytes]) -> list[str]:
    events = [_event("tts.connection.done"), _event("tts.response.created")]
    events.append(_event("tts.response.sentence.start", text="Hello"))
    for chunk in pcm_chunks:
        events.append(
            _event(
                "tts.response.audio.delta",
                status="unfinished",
                audio=base64.b64encode(chunk).decode(),
                duration=0.01,
            )
        )
    events.append(_event("tts.response.sentence.end", text="Hello"))
    events.append(_event("tts.response.audio.done", audio=""))
    return events


@pytest.mark.asyncio
async def test_stepfun_tts_happy_path_and_wire_shape(fake_settings: Settings) -> None:
    ws = FakeWebSocket(_session_events([make_pcm_bytes(240), make_pcm_bytes(240)]))
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return ws

    provider = StepfunTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)
    times = iter([0.0, 0.1])

    with (
        patch(
            "coval_bench.providers.tts.stepfun.time.monotonic",
            side_effect=lambda: next(times, 10.0),
        ),
        patch(_CONNECT, side_effect=connect_side_effect),
    ):
        result = await provider.synthesize("Hello from StepFun")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms == pytest.approx(100.0)
    assert (result.provider, result.model, result.voice) == ("stepfun", _MODEL, _VOICE)
    assert result.audio_path is not None
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    result.audio_path.unlink()

    assert captured["url"] == f"wss://api.stepfun.ai/v1/realtime/audio?model={_MODEL}"
    assert captured["kwargs"] == {
        "additional_headers": {"Authorization": "Bearer test-stepfun-key"},
        "max_size": 16 * 1024 * 1024,
    }
    sent = [json.loads(m) for m in ws.sent if isinstance(m, str)]
    assert [m["type"] for m in sent] == ["tts.create", "tts.text.delta", "tts.text.done"]
    assert sent[0]["data"] == {
        "session_id": _SESSION,
        "voice_id": _VOICE,
        "language": "en",
        "response_format": "pcm",
        "sample_rate": 24000,
        "mode": "sentence",
    }
    assert sent[1]["data"] == {"session_id": _SESSION, "text": "Hello from StepFun"}
    assert sent[2]["data"] == {"session_id": _SESSION}


@pytest.mark.asyncio
async def test_stepfun_tts_error_event(fake_settings: Settings) -> None:
    ws = FakeWebSocket(
        [
            _event("tts.connection.done"),
            _event("tts.response.created"),
            _event(
                "tts.response.error",
                code="503",
                message="The engine is currently overloaded, please try again later",
            ),
        ]
    )
    provider = StepfunTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    with patch(_CONNECT, return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error == "The engine is currently overloaded, please try again later (code=503)"
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_stepfun_tts_close_without_audio_is_silent_failure(
    fake_settings: Settings,
) -> None:
    ws = FakeWebSocket([_event("tts.connection.done"), _event("tts.response.created")])
    provider = StepfunTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    with patch(_CONNECT, return_value=ws):
        result = await provider.synthesize("Hello")

    assert result.error == "provider closed the stream without sending audio or an error"
    assert result.audio_path is None
    assert result.ttfa_ms is None


def test_stepfun_tts_missing_api_key_raises() -> None:
    settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        log_level="DEBUG",
        stepfun_api_key=None,
    )
    with pytest.raises(ValueError, match="stepfun_api_key is required"):
        StepfunTTSProvider(settings, model=_MODEL, voice=_VOICE)
