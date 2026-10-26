# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.meta (MetaSTTProvider).

The fixture follows the documented Muse Voice Transcribe realtime event shapes.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.stt.meta import MetaSTTProvider
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings

_CONNECT = "coval_bench.providers.stt._streaming.ws_client.connect"


@pytest.mark.asyncio
async def test_meta_success_and_wire_shape(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = MetaSTTProvider(stt_settings)
    sent: list[Any] = []

    with patch(_CONNECT, return_value=fake_connect(load_fixture_events("meta"), sent)) as connect:
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.error is None
    assert result.provider == "meta"
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is not None
    assert result.complete_transcript == "Hello world, how are you?"
    assert result.word_count == 5

    assert connect.call_args.args[0] == "wss://api.meta.ai/v1/asr/realtime"
    # Auth rides the first frame; no Authorization header.
    assert connect.call_args.kwargs.get("additional_headers") is None
    control = [json.loads(m) for m in sent if isinstance(m, str)]
    assert control == [
        {
            "authorization": {"accessToken": "Bearer test-api-key-00000000000000000000"},
            "audioEncoding": "PCM_16KHZ",
            "model": "muse-voice-transcribe-1.0",
            "mode": "PUSH_TO_TALK",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
            "languageBias": ["English"],
        },
        {"type": "endStream"},
    ]
    assert b"".join(m for m in sent if isinstance(m, bytes)) == audio_pcm_bytes


@pytest.mark.asyncio
async def test_meta_without_final_keeps_partial_and_fails_the_clip(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    events = [e for e in load_fixture_events("meta") if not e.get("final")]
    provider = MetaSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.complete_transcript == "Hello world, how are"
    assert result.audio_to_final_seconds is None
    assert result.error == "Muse stream closed without a final transcript"


@pytest.mark.asyncio
async def test_meta_rejected_handshake_sets_error(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [{"type": "error", "message": "invalid access token"}]
    provider = MetaSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.error == "Muse handshake rejected: invalid access token"
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_meta_stream_error_event_sets_error(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [
        {"sessionId": "s1"},
        {"type": "error", "message": "backlog exceeded", "sessionId": "s1"},
    ]
    provider = MetaSTTProvider(stt_settings)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.error == "backlog exceeded"


@pytest.mark.asyncio
async def test_meta_rejects_unsupported_audio(stt_settings: Settings) -> None:
    provider = MetaSTTProvider(stt_settings)

    wrong_rate = await provider.measure_ttft(b"\x00\x00", 1, 2, 8000)
    stereo = await provider.measure_ttft(b"\x00\x00", 2, 2, 16000)

    assert wrong_rate.error == "Muse Voice Transcribe needs 16 or 24 kHz PCM; got 8000 Hz"
    assert stereo.error is not None and "mono 16-bit" in stereo.error


def test_meta_construction_guard() -> None:
    with pytest.raises(ValueError, match="meta_api_key is required"):
        MetaSTTProvider(make_settings(meta_api_key=None))
