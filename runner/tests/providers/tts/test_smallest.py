# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Smallest AI TTS provider."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.smallest import SmallestTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_msg(pcm: bytes) -> str:
    """Build a JSON chunk envelope with base64-encoded PCM."""
    return json.dumps({"status": "chunk", "data": {"audio": base64.b64encode(pcm).decode()}})


def _done_msg() -> str:
    return json.dumps({"status": "complete", "message": "All chunks sent", "done": True})


def _make_ws(pcm_chunks: list[bytes]) -> FakeWebSocket:
    """Return a FakeWebSocket that yields chunk envelopes then a done message."""
    messages = [_chunk_msg(c) for c in pcm_chunks] + [_done_msg()]
    return FakeWebSocket(messages)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smallest_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    pcm = make_pcm_bytes(480)
    fake_ws = _make_ws([pcm, pcm, pcm, pcm])

    with patch("coval_bench.providers.tts.smallest.ws_client.connect", return_value=fake_ws):
        provider = SmallestTTSProvider(fake_settings, model="lightning_v3.1_pro", voice="kaitlyn")
        result = await provider.synthesize("Hello from Smallest AI")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 10_000
    assert result.audio_path is not None and result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "smallest"
    assert result.model == "lightning_v3.1_pro"

    # Confirm the payload sent to the WebSocket was correct.
    assert len(fake_ws.sent) == 1
    payload = json.loads(fake_ws.sent[0])
    assert payload["text"] == "Hello from Smallest AI"
    assert payload["voice_id"] == "kaitlyn"
    assert payload["model"] == "lightning_v3.1_pro"
    assert payload["sample_rate"] == 24000
    assert payload["language"] == "en"

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_smallest_all_models(fake_settings: Settings) -> None:
    for model in SmallestTTSProvider._VALID_MODELS:
        fake_ws = _make_ws([make_pcm_bytes(240)])
        with patch("coval_bench.providers.tts.smallest.ws_client.connect", return_value=fake_ws):
            provider = SmallestTTSProvider(fake_settings, model=model, voice="kaitlyn")
            result = await provider.synthesize("test")
        assert result.error is None, f"{model}: {result.error}"
        if result.audio_path is not None:
            result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smallest_empty_audio(fake_settings: Settings) -> None:
    """Server sends complete with no chunks — no audio_path, no ttfa."""
    fake_ws = FakeWebSocket([_done_msg()])

    with patch("coval_bench.providers.tts.smallest.ws_client.connect", return_value=fake_ws):
        provider = SmallestTTSProvider(fake_settings, model="lightning_v3.1_pro", voice="kaitlyn")
        result = await provider.synthesize("silence")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_smallest_done_via_done_flag(fake_settings: Settings) -> None:
    """done=true on a non-complete status still terminates the loop."""
    msg = json.dumps({"status": "other", "done": True})
    fake_ws = FakeWebSocket([_chunk_msg(make_pcm_bytes(240)), msg])

    with patch("coval_bench.providers.tts.smallest.ws_client.connect", return_value=fake_ws):
        provider = SmallestTTSProvider(fake_settings, model="lightning_v3.1_pro", voice="kaitlyn")
        result = await provider.synthesize("test")

    assert result.error is None
    assert result.ttfa_ms is not None
    if result.audio_path is not None:
        result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smallest_connection_error(fake_settings: Settings) -> None:
    with patch(
        "coval_bench.providers.tts.smallest.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        provider = SmallestTTSProvider(fake_settings, model="lightning_v3.1_pro", voice="kaitlyn")
        result = await provider.synthesize("hi")

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_smallest_name_and_model(fake_settings: Settings) -> None:
    p = SmallestTTSProvider(fake_settings, model="lightning_v3.1_pro", voice="kaitlyn")
    assert p.name == "smallest-lightning_v3.1_pro"
    assert p.model == "lightning_v3.1_pro"


def test_smallest_rejects_unsupported_model(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unsupported Smallest AI model"):
        SmallestTTSProvider(fake_settings, model="not-a-real-model", voice="kaitlyn")


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_smallest_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        smallest_api_key=None,
    )
    with pytest.raises(ValueError, match="smallest_api_key"):
        SmallestTTSProvider(settings_no_key, model="lightning_v3.1_pro", voice="kaitlyn")
