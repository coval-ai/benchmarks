# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.gemini (GeminiSTTProvider).

All tests use FakeWebSocket — no live network calls are made. The event
fixtures mirror the Gemini Live API wire protocol as observed live: a
``setupComplete`` handshake, then ``serverContent`` frames carrying
``interimInputTranscription`` (speculative) and ``inputTranscription`` (final)
segments, with ``generationComplete`` per utterance — the server never ends
the session, so the client closes after the post-stream-end completion. The
live API delivers JSON in binary frames, so decoding ``bytes`` events is
pinned explicitly.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.gemini import GeminiSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events

_CONNECT = "coval_bench.providers.stt.gemini.ws_client.connect"


def _fake_connect(events: list[Any]) -> tuple[Any, FakeWebSocket]:
    ws = FakeWebSocket(events)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, ws


async def _measure(
    provider: GeminiSTTProvider, events: list[Any], audio: bytes
) -> tuple[Any, FakeWebSocket]:
    cm, ws = _fake_connect(events)
    with patch(_CONNECT, return_value=cm):
        result = await provider.measure_ttft(
            audio_data=audio,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )
    return result, ws


@pytest.mark.asyncio
async def test_gemini_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events = load_fixture_events("gemini", "events-success")
    provider = GeminiSTTProvider(api_key=fake_api_key)

    result, ws = await _measure(provider, events, audio_pcm_bytes)

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    assert result.audio_to_final_seconds is not None
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)

    setup = json.loads(ws._sent[0])["setup"]
    assert setup["model"] == "models/gemini-3.5-transcribe-live"
    # SMART mode cleans disfluencies, which would skew WER against verbatim references.
    assert setup["inputAudioTranscription"]["mode"] == "VERBATIM"
    assert json.loads(ws._sent[-1]) == {"realtimeInput": {"audioStreamEnd": True}}


@pytest.mark.asyncio
async def test_gemini_decodes_binary_frames(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """The live API sends JSON in binary frames; they must decode, not be skipped."""
    events: list[Any] = [
        json.dumps(event).encode("utf-8")
        for event in load_fixture_events("gemini", "events-success")
    ]
    provider = GeminiSTTProvider(api_key=fake_api_key)

    result, _ = await _measure(provider, events, audio_pcm_bytes)

    assert result.error is None
    assert result.complete_transcript == "hello world how are you"
    assert result.ttft_seconds is not None


@pytest.mark.asyncio
async def test_gemini_stream_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [
        {"setupComplete": {}},
        {"error": {"message": "quota exceeded"}},
    ]
    provider = GeminiSTTProvider(api_key=fake_api_key)

    result, _ = await _measure(provider, events, audio_pcm_bytes)

    assert result.error == "quota exceeded"
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_gemini_setup_error(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [{"error": {"message": "invalid model"}}]
    provider = GeminiSTTProvider(api_key=fake_api_key)

    result, _ = await _measure(provider, events, audio_pcm_bytes)

    assert result.error is not None
    assert "invalid model" in result.error
    assert result.ttft_seconds is None
