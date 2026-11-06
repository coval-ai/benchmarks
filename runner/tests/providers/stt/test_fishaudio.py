# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.fishaudio (FishAudioSTTProvider)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import ormsgpack
import pytest

from coval_bench.config import Settings
from coval_bench.providers.stt.fishaudio import FishAudioSTTProvider
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings

_CONNECT = "coval_bench.providers.stt._streaming.ws_client.connect"
_MODEL = "transcribe-stream-1"
_STARTED = {"type": "session_started", "seq": 0, "session_id": "s-test", "request_id": "r-test"}


def _frames(events: list[dict[str, Any]]) -> list[bytes]:
    return [ormsgpack.packb(e) for e in [_STARTED, *events]]


@pytest.mark.asyncio
async def test_fishaudio_success_and_wire_shape(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    provider = FishAudioSTTProvider(stt_settings, model=_MODEL)
    sent: list[Any] = []

    with patch(
        _CONNECT, return_value=fake_connect(_frames(load_fixture_events("fishaudio")), sent)
    ) as connect:
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.error is None
    assert result.provider == f"fishaudio-{_MODEL}"
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is not None
    assert result.vad_events_count == 1
    assert result.complete_transcript == "Hello world, how are you?"

    assert connect.call_args.args[0] == "wss://api.fish.audio/v1/asr/realtime"
    assert connect.call_args.kwargs["additional_headers"] == {
        "Authorization": "Bearer test-api-key-00000000000000000000",
        "model": _MODEL,
    }
    msgs = [ormsgpack.unpackb(m) for m in sent]
    assert msgs[0] == {
        "type": "config",
        "audio": {"format": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        "turn_detection": {"mode": "manual"},
        "results": {"drafts": True, "diarize": False, "tag_audio_events": False},
    }
    assert [m["type"] for m in msgs[-2:]] == ["finalize", "audio_done"]
    audio = [m["data"] for m in msgs if m["type"] == "audio"]
    assert b"".join(audio) == audio_pcm_bytes


@pytest.mark.asyncio
async def test_fishaudio_fatal_error_fails_the_clip(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    error = {
        "type": "error",
        "seq": 0,
        "code": 400,
        "error_type": "invalid_config",
        "message": "bad field",
        "recoverable": False,
    }
    provider = FishAudioSTTProvider(stt_settings, model=_MODEL)

    with patch(_CONNECT, return_value=fake_connect([ormsgpack.packb(error)])):
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.error == "bad field"
    assert result.complete_transcript is None


def test_fishaudio_construction_guards() -> None:
    with pytest.raises(ValueError, match="fishaudio_api_key is required"):
        FishAudioSTTProvider(make_settings(fishaudio_api_key=None), model=_MODEL)
