# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rime TTS provider — WebSocket /ws3 JSON endpoint."""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.rime import SAMPLE_RATE, RimeTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_connect(ws: FakeWebSocket) -> Any:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _ws3_events(pcm_chunks: list[bytes]) -> list[str]:
    """Build a /ws3 event stream: chunk messages followed by done."""
    events = [
        json.dumps({"type": "chunk", "data": base64.b64encode(c).decode()}) for c in pcm_chunks
    ]
    events.append(json.dumps({"type": "done"}))
    return events


# ---------------------------------------------------------------------------
# Happy path — all three valid models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["arcana", "mistv3", "coda"])
async def test_rime_happy_path(fake_settings: Settings, model: str) -> None:
    """All valid models: ttfa set, WAV written with RIFF header."""
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_ws3_events(chunks))
    provider = RimeTTSProvider(fake_settings, model=model, voice="luna")

    with patch(
        "coval_bench.providers.tts.rime.ws_client.connect",
        return_value=_fake_connect(ws),
    ):
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
# Protocol: sent messages and URL shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_sends_text_and_eos(fake_settings: Settings) -> None:
    """Provider sends {"text": …} then {"operation": "eos"} — in that order."""
    ws = FakeWebSocket(_ws3_events([make_pcm_bytes(240)]))
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch(
        "coval_bench.providers.tts.rime.ws_client.connect",
        return_value=_fake_connect(ws),
    ):
        await provider.synthesize("test text")

    assert len(ws.sent) >= 2
    first = json.loads(ws.sent[0])
    second = json.loads(ws.sent[1])
    assert first == {"text": "test text"}
    assert second == {"operation": "eos"}


@pytest.mark.asyncio
async def test_rime_url_shape(fake_settings: Settings) -> None:
    """WS URL must target users-ws.rime.ai /ws3 with correct query params."""
    ws = FakeWebSocket(_ws3_events([make_pcm_bytes(240)]))
    captured: dict[str, str] = {}

    def _capture(url: str, **_kwargs: object) -> Any:
        captured["url"] = url
        return _fake_connect(ws)

    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch(
        "coval_bench.providers.tts.rime.ws_client.connect",
        side_effect=_capture,
    ):
        await provider.synthesize("url test")

    parsed = urlparse(captured["url"])
    qs = parse_qs(parsed.query)

    assert parsed.hostname == "users-ws.rime.ai"
    assert parsed.path == "/ws3"
    assert qs["modelId"] == ["coda"]
    assert qs["audioFormat"] == ["pcm"]
    assert qs["samplingRate"] == [str(SAMPLE_RATE)]
    assert qs["segment"] == ["never"]
    assert qs["speaker"] == ["luna"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["arcana", "mistv3"])
async def test_rime_url_model_id(fake_settings: Settings, model: str) -> None:
    """modelId in the URL matches the requested model for arcana and mistv3."""
    ws = FakeWebSocket(_ws3_events([make_pcm_bytes(240)]))
    captured: dict[str, str] = {}

    def _capture(url: str, **_kwargs: object) -> Any:
        captured["url"] = url
        return _fake_connect(ws)

    provider = RimeTTSProvider(fake_settings, model=model, voice="luna")

    with patch(
        "coval_bench.providers.tts.rime.ws_client.connect",
        side_effect=_capture,
    ):
        await provider.synthesize("test")

    assert parse_qs(urlparse(captured["url"]).query)["modelId"] == [model]


# ---------------------------------------------------------------------------
# Protocol: timestamps event is silently ignored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_timestamps_ignored(fake_settings: Settings) -> None:
    """Interleaved timestamps events don't interrupt audio assembly."""
    pcm = make_pcm_bytes(240)
    timestamps_event = json.dumps(
        {
            "type": "timestamps",
            "word_timestamps": {
                "words": ["Hello"],
                "start": [0.0],
                "end": [0.36],
            },
        }
    )
    events = [
        json.dumps({"type": "chunk", "data": base64.b64encode(pcm).decode()}),
        timestamps_event,
        json.dumps({"type": "chunk", "data": base64.b64encode(pcm).decode()}),
        json.dumps({"type": "done"}),
    ]
    ws = FakeWebSocket(events)
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch(
        "coval_bench.providers.tts.rime.ws_client.connect",
        return_value=_fake_connect(ws),
    ):
        result = await provider.synthesize("Hello")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    # Both PCM chunks assembled: 2 × 240 frames × 2 bytes = 960 bytes of PCM
    data = result.audio_path.read_bytes()
    assert len(data) > 960
    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_server_error_event(fake_settings: Settings) -> None:
    """{"type":"error"} frame from server → result.error populated."""
    events = [json.dumps({"type": "error", "message": "rate limit exceeded"})]
    ws = FakeWebSocket(events)
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch(
        "coval_bench.providers.tts.rime.ws_client.connect",
        return_value=_fake_connect(ws),
    ):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "rate limit exceeded" in result.error
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_rime_connect_failure(fake_settings: Settings) -> None:
    """WebSocket connect OSError → result.error populated, audio_path None."""
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch(
        "coval_bench.providers.tts.rime.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        result = await provider.synthesize("error test")

    assert result.error is not None
    assert "refused" in result.error.lower()
    assert result.audio_path is None


@pytest.mark.asyncio
async def test_rime_empty_response(fake_settings: Settings) -> None:
    """done with no chunks → audio_path None, ttfa_ms None, no error."""
    ws = FakeWebSocket([json.dumps({"type": "done"})])
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch(
        "coval_bench.providers.tts.rime.ws_client.connect",
        return_value=_fake_connect(ws),
    ):
        result = await provider.synthesize("silent test")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# TTFA correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_ttfa_set_on_first_chunk_only(fake_settings: Settings) -> None:
    """ttfa_ms is measured at the first chunk and not overwritten by later chunks."""
    chunks = [make_pcm_bytes(240), make_pcm_bytes(240), make_pcm_bytes(240)]
    ws = FakeWebSocket(_ws3_events(chunks))
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    # clock: start=0.0, chunk-1=0.1s, chunk-2=0.5s, chunk-3=0.9s.
    # If the `if ttfa_ms is None` guard is removed, chunks 2 and 3 overwrite to
    # 500 ms and 900 ms respectively, and the assertion below fails.
    clock = iter([0.0, 0.1, 0.5, 0.9])

    with (
        patch("coval_bench.providers.tts.rime.ws_client.connect", return_value=_fake_connect(ws)),
        patch("coval_bench.providers.tts.rime.time.monotonic", side_effect=clock),
    ):
        result = await provider.synthesize("three chunks")

    assert result.error is None
    assert result.ttfa_ms == pytest.approx(100.0)
    if result.audio_path:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_rime_auth_header(fake_settings: Settings) -> None:
    """Bearer token is passed in additional_headers when opening the WS connection."""
    ws = FakeWebSocket(_ws3_events([make_pcm_bytes(240)]))
    captured: dict[str, str] = {}

    def _capture(url: str, additional_headers: dict[str, str] | None = None, **_kw: object) -> Any:
        if additional_headers:
            captured.update(additional_headers)
        return _fake_connect(ws)

    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch("coval_bench.providers.tts.rime.ws_client.connect", side_effect=_capture):
        await provider.synthesize("auth test")

    assert captured.get("Authorization") == "Bearer test-rime-key"


@pytest.mark.asyncio
async def test_rime_empty_chunk_not_counted(fake_settings: Settings) -> None:
    """A chunk event whose base64 data decodes to empty bytes must not set ttfa_ms or audio."""
    events = [
        json.dumps({"type": "chunk", "data": base64.b64encode(b"").decode()}),
        json.dumps({"type": "done"}),
    ]
    ws = FakeWebSocket(events)
    provider = RimeTTSProvider(fake_settings, model="coda", voice="luna")

    with patch("coval_bench.providers.tts.rime.ws_client.connect", return_value=_fake_connect(ws)):
        result = await provider.synthesize("test")

    assert result.error is None
    # If the `if audio_bytes:` guard is removed, audio_chunks = [b""] is truthy
    # so _write_wav is called and ttfa_ms is set — both assertions below would fail.
    assert result.ttfa_ms is None
    assert result.audio_path is None


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_invalid_model(fake_settings: Settings) -> None:
    """Unknown model returns error result without opening a WS connection."""
    provider = RimeTTSProvider(fake_settings, model="mistv4", voice="luna")
    result = await provider.synthesize("test")
    assert result.error is not None
    assert "Unsupported" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# Voice fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_voice_fallback(fake_settings: Settings) -> None:
    """voice=None falls back to 'luna' in the WS query string."""
    ws = FakeWebSocket(_ws3_events([make_pcm_bytes(240)]))
    captured: dict[str, str] = {}

    def _capture(url: str, **_kwargs: object) -> Any:
        captured["url"] = url
        return _fake_connect(ws)

    provider = RimeTTSProvider(fake_settings, model="coda", voice=None)

    with patch(
        "coval_bench.providers.tts.rime.ws_client.connect",
        side_effect=_capture,
    ):
        await provider.synthesize("test")

    assert parse_qs(urlparse(captured["url"]).query)["speaker"] == ["luna"]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["arcana", "mistv3", "coda"])
def test_rime_name_and_model(fake_settings: Settings, model: str) -> None:
    p = RimeTTSProvider(fake_settings, model=model, voice="luna")
    assert p.name == f"rime-{model}"
    assert p.model == model


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_rime_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",  # type: ignore[arg-type]
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        rime_api_key=None,
    )
    with pytest.raises(ValueError, match="rime_api_key"):
        RimeTTSProvider(settings_no_key, model="coda", voice="luna")
