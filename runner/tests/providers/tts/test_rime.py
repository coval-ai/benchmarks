# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rime TTS provider — Coda v1 WebSocket and the legacy /ws3 endpoint."""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.rime import _MODEL_SAMPLE_RATES, RimeTTSProvider
from tests.providers.fakes import FakeWebSocket, fake_connect

from .conftest import make_pcm_bytes

_CONNECT = "coval_bench.providers.tts.rime.ws_client.connect"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ws3_events(pcm_chunks: list[bytes]) -> list[str]:
    """Build a legacy /ws3 event stream: chunk messages followed by done."""
    events = [
        json.dumps({"type": "chunk", "data": base64.b64encode(c).decode()}) for c in pcm_chunks
    ]
    events.append(json.dumps({"type": "done"}))
    return events


def _coda_events(pcm_chunks: list[bytes], *, ready: bool = True) -> list[str]:
    """Build a Coda v1 stream: ready, started, audio messages, done."""
    events: list[str] = []
    if ready:
        events.append(json.dumps({"ready": {"protocol": 1, "languages": ["en"]}}))
    events.append(json.dumps({"contextId": "turn-1", "started": {"requestId": "r-1"}}))
    events.extend(
        json.dumps({"contextId": "turn-1", "audio": base64.b64encode(c).decode()})
        for c in pcm_chunks
    )
    events.append(json.dumps({"contextId": "turn-1", "done": {}}))
    return events


def _events_for(model: str, pcm_chunks: list[bytes]) -> list[str]:
    return _coda_events(pcm_chunks) if model == "coda" else _ws3_events(pcm_chunks)


def _capture_connect(ws: FakeWebSocket, captured: dict[str, Any]) -> Any:
    def _connect(url: str, additional_headers: dict[str, str] | None = None, **_kw: object) -> Any:
        captured["url"] = url
        captured["headers"] = additional_headers or {}
        return fake_connect(ws)

    return _connect


# ---------------------------------------------------------------------------
# Happy path — all three valid models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["mistv3", "coda"])
async def test_rime_happy_path(fake_settings: Settings, model: str) -> None:
    """All valid models: ttfa set, WAV written with RIFF header."""
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_events_for(model, chunks))
    provider = RimeTTSProvider(fake_settings, model=model, voice="luna")

    with patch(_CONNECT, return_value=fake_connect(ws)):
        result = await provider.synthesize("Hello from Rime")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.ttfa_ms is not None
    assert 0 < result.ttfa_ms < 60_000
    assert result.provider == "rime"
    assert result.model == model
    assert result.voice == "luna"
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Coda: coda.api.rime.ai v1 protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_coda_url_and_start_message(fake_settings: Settings) -> None:
    """Coda connects to coda.api.rime.ai/ws and sends a single start with the full text."""
    ws = FakeWebSocket(_coda_events([make_pcm_bytes(240)]))
    captured: dict[str, Any] = {}
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch(_CONNECT, side_effect=_capture_connect(ws, captured)):
        result = await provider.synthesize("coda test")

    assert result.error is None
    parsed = urlparse(captured["url"])
    assert parsed.hostname == "coda.api.rime.ai"
    assert parsed.path == "/ws"
    assert parsed.query == ""
    assert captured["headers"].get("Authorization") == "Bearer test-rime-key"

    assert len(ws.sent) == 1
    start = json.loads(ws.sent[0])
    assert start["contextId"]
    assert start["start"]["text"] == "coda test"
    assert start["start"]["speaker"] == "luna"
    assert start["start"]["audioParameters"] == {
        "audioFormat": "audio/pcm",
        "samplingRate": _MODEL_SAMPLE_RATES["coda"],
    }
    if result.audio_path:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_rime_coda_missing_ready(fake_settings: Settings) -> None:
    """A first frame that isn't ready is an error, not silently treated as audio."""
    ws = FakeWebSocket(_coda_events([make_pcm_bytes(240)], ready=False))
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch(_CONNECT, return_value=fake_connect(ws)):
        result = await provider.synthesize("no ready")

    assert result.error is not None
    assert "ready" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_rime_coda_server_error(fake_settings: Settings) -> None:
    """A Coda error frame populates result.error with the server message."""
    events = [
        json.dumps({"ready": {"protocol": 1}}),
        json.dumps(
            {
                "contextId": "turn-1",
                "error": {"kind": "invalid_input", "message": "unknown speaker"},
            }
        ),
    ]
    ws = FakeWebSocket(events)
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch(_CONNECT, return_value=fake_connect(ws)):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "unknown speaker" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_rime_coda_ttfa_set_on_first_audio_only(fake_settings: Settings) -> None:
    """ttfa_ms is stamped on the first audio frame and not moved by later ones."""
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_coda_events(chunks))
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    clock = iter([0.0, 0.1, 0.5, 0.9])

    with (
        patch(_CONNECT, return_value=fake_connect(ws)),
        patch("coval_bench.providers.tts.rime.time.monotonic", side_effect=clock),
    ):
        result = await provider.synthesize("three chunks")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    if result.audio_path:
        result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Legacy /ws3: mistv3
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_legacy_sends_text_and_eos(fake_settings: Settings) -> None:
    """Legacy models send {"text": …} then {"operation": "eos"} — in that order."""
    ws = FakeWebSocket(_ws3_events([make_pcm_bytes(240)]))
    provider = RimeTTSProvider(fake_settings, model="mistv3", voice="luna")

    with patch(_CONNECT, return_value=fake_connect(ws)):
        await provider.synthesize("test text")

    assert len(ws.sent) >= 2
    assert json.loads(ws.sent[0]) == {"text": "test text"}
    assert json.loads(ws.sent[1]) == {"operation": "eos"}


@pytest.mark.asyncio
async def test_rime_legacy_url_shape(fake_settings: Settings) -> None:
    """mistv3 targets users-ws.rime.ai /ws3 with its native 22050 Hz sample rate."""
    model = "mistv3"
    ws = FakeWebSocket(_ws3_events([make_pcm_bytes(240)]))
    captured: dict[str, Any] = {}
    provider = RimeTTSProvider(fake_settings, model=model, voice="luna")

    with patch(_CONNECT, side_effect=_capture_connect(ws, captured)):
        result = await provider.synthesize("url test")

    assert result.error is None
    parsed = urlparse(captured["url"])
    qs = parse_qs(parsed.query)

    assert parsed.hostname == "users-ws.rime.ai"
    assert parsed.path == "/ws3"
    assert qs["modelId"] == [model]
    assert qs["audioFormat"] == ["pcm"]
    assert qs["samplingRate"] == [str(_MODEL_SAMPLE_RATES[model])]
    assert qs["segment"] == ["never"]
    assert qs["speaker"] == ["luna"]
    assert captured["headers"].get("Authorization") == "Bearer test-rime-key"
    if result.audio_path:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_rime_legacy_timestamps_ignored(fake_settings: Settings) -> None:
    """Interleaved timestamps events don't interrupt audio assembly."""
    pcm = make_pcm_bytes(240)
    events = [
        json.dumps({"type": "chunk", "data": base64.b64encode(pcm).decode()}),
        json.dumps({"type": "timestamps", "word_timestamps": {"words": ["Hello"]}}),
        json.dumps({"type": "chunk", "data": base64.b64encode(pcm).decode()}),
        json.dumps({"type": "done"}),
    ]
    ws = FakeWebSocket(events)
    provider = RimeTTSProvider(fake_settings, model="mistv3", voice="luna")

    with patch(_CONNECT, return_value=fake_connect(ws)):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.audio_path is not None
    assert len(result.audio_path.read_bytes()) > 960
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_rime_legacy_server_error_event(fake_settings: Settings) -> None:
    """{"type":"error"} frame from server → result.error populated."""
    ws = FakeWebSocket([json.dumps({"type": "error", "message": "rate limit exceeded"})])
    provider = RimeTTSProvider(fake_settings, model="mistv3", voice="luna")

    with patch(_CONNECT, return_value=fake_connect(ws)):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "rate limit exceeded" in result.error
    assert result.audio_path is None


# ---------------------------------------------------------------------------
# Shared behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["mistv3", "coda"])
async def test_rime_connect_failure(fake_settings: Settings, model: str) -> None:
    """WebSocket connect OSError → result.error populated, audio_path None."""
    provider = RimeTTSProvider(fake_settings, model=model, voice="luna")

    with patch(_CONNECT, side_effect=OSError("connection refused")):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "refused" in result.error.lower()
    assert result.audio_path is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["mistv3", "coda"])
async def test_rime_empty_response(fake_settings: Settings, model: str) -> None:
    """done with no audio → audio_path None, ttfa_ms None, stream-closed error."""
    ws = FakeWebSocket(_events_for(model, []))
    provider = RimeTTSProvider(fake_settings, model=model, voice="luna")

    with patch(_CONNECT, return_value=fake_connect(ws)):
        result = await provider.synthesize("silent test")

    assert result.error == ("provider closed the stream without sending audio or an error")
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["mistv3", "coda"])
async def test_rime_empty_chunk_not_counted(fake_settings: Settings, model: str) -> None:
    """An audio frame that decodes to empty bytes must not set ttfa_ms or audio."""
    ws = FakeWebSocket(_events_for(model, [b""]))
    provider = RimeTTSProvider(fake_settings, model=model, voice="luna")

    with patch(_CONNECT, return_value=fake_connect(ws)):
        result = await provider.synthesize("test")

    assert result.error == ("provider closed the stream without sending audio or an error")
    assert result.ttfa_ms is None
    assert result.audio_path is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["mistv3", "coda"])
async def test_rime_voice_fallback(fake_settings: Settings, model: str) -> None:
    """voice=None falls back to 'luna' on both endpoints."""
    ws = FakeWebSocket(_events_for(model, [make_pcm_bytes(240)]))
    captured: dict[str, Any] = {}
    provider = RimeTTSProvider(fake_settings, model=model, voice=None)

    with patch(_CONNECT, side_effect=_capture_connect(ws, captured)):
        result = await provider.synthesize("test")

    if model == "coda":
        assert json.loads(ws.sent[0])["start"]["speaker"] == "luna"
    else:
        assert parse_qs(urlparse(captured["url"]).query)["speaker"] == ["luna"]
    if result.audio_path:
        result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["mistv3", "coda"])
def test_rime_name_and_model(fake_settings: Settings, model: str) -> None:
    p = RimeTTSProvider(fake_settings, model=model, voice="luna")
    assert p.name == f"rime-{model}"
    assert p.model == model


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_rime_missing_api_key() -> None:
    settings_no_key = Settings(
        rime_api_key=None,
    )
    with pytest.raises(ValueError, match="rime_api_key"):
        RimeTTSProvider(settings_no_key, model="coda", voice="luna")
