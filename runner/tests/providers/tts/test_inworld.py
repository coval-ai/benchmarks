# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Inworld AI TTS provider."""

from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts.inworld import InworldTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_msg(pcm: bytes) -> str:
    """Build a JSON audioChunk envelope with base64-encoded PCM."""
    return json.dumps(
        {
            "result": {
                "contextId": "coval-bench",
                "audioChunk": {"audioContent": base64.b64encode(pcm).decode()},
            }
        }
    )


def _done_msg() -> str:
    return json.dumps({"result": {"contextId": "coval-bench", "flushCompleted": {}}})


def _make_ws(pcm_chunks: list[bytes]) -> FakeWebSocket:
    """Return a FakeWebSocket that yields audio chunks then a flush-completed message."""
    messages = [_chunk_msg(c) for c in pcm_chunks] + [_done_msg()]
    return FakeWebSocket(messages)


def _wav_wrap(pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw PCM in a standalone LINEAR16 WAV, as Inworld sends each chunk."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inworld_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    pcm = make_pcm_bytes(480)
    fake_ws = _make_ws([pcm, pcm, pcm, pcm])

    with patch("coval_bench.providers.tts.inworld.ws_client.connect", return_value=fake_ws):
        provider = InworldTTSProvider(fake_settings, model="inworld-tts-1.5-max", voice="Ashley")
        result = await provider.synthesize("Hello from Inworld")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 10_000
    assert result.audio_path is not None and result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "inworld"
    assert result.model == "inworld-tts-1.5-max"

    # Confirm the two messages sent to the WebSocket were correct.
    assert len(fake_ws.sent) == 2
    create = json.loads(fake_ws.sent[0])
    assert create["create"]["voiceId"] == "Ashley"
    assert create["create"]["modelId"] == "inworld-tts-1.5-max"
    assert create["create"]["audioConfig"]["audioEncoding"] == "LINEAR16"
    assert create["create"]["audioConfig"]["sampleRateHertz"] == 24000
    send_text = json.loads(fake_ws.sent[1])
    assert send_text["send_text"]["text"] == "Hello from Inworld"
    assert "flush_context" in send_text["send_text"]

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_inworld_strips_per_chunk_wav_header(fake_settings: Settings, tmp_path: Path) -> None:
    """Inworld wraps every chunk in its own WAV; the saved audio must be raw PCM.

    Without stripping, the embedded RIFF headers read as audible at t=0 (corrupting
    TTFA) and land as clicks inside the assembled audio (corrupting WER).
    """
    pcm = make_pcm_bytes(480)
    fake_ws = _make_ws([_wav_wrap(pcm), _wav_wrap(pcm), _wav_wrap(pcm)])

    with patch("coval_bench.providers.tts.inworld.ws_client.connect", return_value=fake_ws):
        provider = InworldTTSProvider(fake_settings, model="inworld-tts-1.5-max", voice="Ashley")
        result = await provider.synthesize("header test")

    assert result.error is None, result.error
    assert result.audio_path is not None
    with wave.open(str(result.audio_path), "rb") as w:
        written = w.readframes(w.getnframes())
    # Exactly the three raw PCM payloads, with no RIFF headers embedded.
    assert written == pcm * 3
    assert b"RIFF" not in written

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_inworld_auth_in_connect_url(fake_settings: Settings) -> None:
    """The key goes in the URL verbatim, only percent-encoded (not re-encoded).

    Uses a key with base64 reserved chars (+ / =) — what real Inworld keys
    contain — so the encoding is actually exercised.
    """
    settings = fake_settings.model_copy(update={"inworld_api_key": SecretStr("ab+c/d==")})
    fake_ws = _make_ws([make_pcm_bytes(240)])
    with patch(
        "coval_bench.providers.tts.inworld.ws_client.connect", return_value=fake_ws
    ) as connect:
        provider = InworldTTSProvider(settings, model="inworld-tts-1.5-max", voice="Ashley")
        result = await provider.synthesize("hi")

    url = connect.call_args.args[0]
    assert url == (
        "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"
        "?authorization=Basic%20ab%2Bc%2Fd%3D%3D"
    )

    if result.audio_path is not None:
        result.audio_path.unlink()


@pytest.mark.asyncio
async def test_inworld_redacts_key_in_error(fake_settings: Settings) -> None:
    """A connection error that echoes the auth URL must not leak the credential."""
    settings = fake_settings.model_copy(update={"inworld_api_key": SecretStr("ab+c/d==")})
    leaky = (
        "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"
        "?authorization=Basic%20ab%2Bc%2Fd%3D%3D rejected"
    )
    with patch(
        "coval_bench.providers.tts.inworld.ws_client.connect",
        side_effect=RuntimeError(leaky),
    ):
        provider = InworldTTSProvider(settings, model="inworld-tts-1.5-max", voice="Ashley")
        result = await provider.synthesize("hi")

    assert result.error is not None
    assert "ab%2Bc%2Fd%3D%3D" not in result.error
    assert "authorization=***" in result.error


@pytest.mark.asyncio
async def test_inworld_all_models(fake_settings: Settings) -> None:
    for model in InworldTTSProvider._VALID_MODELS:
        fake_ws = _make_ws([make_pcm_bytes(240)])
        with patch("coval_bench.providers.tts.inworld.ws_client.connect", return_value=fake_ws):
            provider = InworldTTSProvider(fake_settings, model=model, voice="Ashley")
            result = await provider.synthesize("test")
        assert result.error is None, f"{model}: {result.error}"
        if result.audio_path is not None:
            result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inworld_empty_audio(fake_settings: Settings) -> None:
    """Server flushes with no chunks — no audio_path, no ttfa."""
    fake_ws = FakeWebSocket([_done_msg()])

    with patch("coval_bench.providers.tts.inworld.ws_client.connect", return_value=fake_ws):
        provider = InworldTTSProvider(fake_settings, model="inworld-tts-1.5-max", voice="Ashley")
        result = await provider.synthesize("silence")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_inworld_error_message(fake_settings: Settings) -> None:
    """A streamed top-level ``error`` ends synthesis as a failure."""
    err = json.dumps({"error": {"code": 13, "message": "internal"}})
    fake_ws = FakeWebSocket([err])

    with patch("coval_bench.providers.tts.inworld.ws_client.connect", return_value=fake_ws):
        provider = InworldTTSProvider(fake_settings, model="inworld-tts-1.5-max", voice="Ashley")
        result = await provider.synthesize("hi")

    assert result.error is not None
    assert "internal" in result.error


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inworld_connection_error(fake_settings: Settings) -> None:
    with patch(
        "coval_bench.providers.tts.inworld.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        provider = InworldTTSProvider(fake_settings, model="inworld-tts-1.5-max", voice="Ashley")
        result = await provider.synthesize("hi")

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_inworld_name_and_model(fake_settings: Settings) -> None:
    p = InworldTTSProvider(fake_settings, model="inworld-tts-1.5-mini", voice="Ashley")
    assert p.name == "inworld-tts-1.5-mini"
    assert p.model == "inworld-tts-1.5-mini"


def test_inworld_rejects_unsupported_model(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unsupported Inworld model"):
        InworldTTSProvider(fake_settings, model="not-a-real-model", voice="Ashley")


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_inworld_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        inworld_api_key=None,
    )
    with pytest.raises(ValueError, match="inworld_api_key"):
        InworldTTSProvider(settings_no_key, model="inworld-tts-1.5-max", voice="Ashley")
