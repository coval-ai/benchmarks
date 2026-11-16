# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.murf (MurfSTTProvider)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.stt.murf import MurfSTTProvider
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings

_CONNECT = "coval_bench.providers.stt._streaming.ws_client.connect"
_MODEL = "finch_global_1.0"


@pytest.mark.asyncio
async def test_murf_success_and_wire_shape(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    provider = MurfSTTProvider(stt_settings, model=_MODEL)
    sent: list[Any] = []

    with patch(_CONNECT, return_value=fake_connect(load_fixture_events("murf"), sent)) as connect:
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.error is None
    assert result.provider == f"murf-{_MODEL}"
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is not None
    assert result.complete_transcript == "Hello world, how are you?"

    assert connect.call_args.args[0] == (
        "wss://us-east.api.murf.ai/v1/transcription/stream-input"
        f"?model={_MODEL}&language=en&commit_mode=manual"
    )
    assert connect.call_args.kwargs["additional_headers"] == {
        "api-key": "test-api-key-00000000000000000000"
    }
    control = [json.loads(m) for m in sent if isinstance(m, str)]
    assert control == [
        {"type": "AudioConfig", "codec": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        {"type": "Finalize"},
        {"type": "CloseStream"},
    ]
    assert b"".join(m for m in sent if isinstance(m, bytes)) == audio_pcm_bytes


@pytest.mark.asyncio
async def test_murf_error_fails_the_clip(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    error = {"type": "Error", "message": "Unauthorized: Invalid or missing API key"}
    provider = MurfSTTProvider(stt_settings, model=_MODEL)

    with patch(_CONNECT, return_value=fake_connect([error])):
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.error == "Unauthorized: Invalid or missing API key"
    assert result.complete_transcript is None


def test_murf_construction_guards() -> None:
    with pytest.raises(ValueError, match="murfai_api_key is required"):
        MurfSTTProvider(make_settings(murfai_api_key=None), model=_MODEL)
