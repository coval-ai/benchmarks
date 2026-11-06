# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.saidwire (SaidwireSTTProvider)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.stt.saidwire import SaidwireSTTProvider
from coval_bench.registries.provider_keys import PROVIDER_ENV
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings

_CONNECT = "coval_bench.providers.stt._streaming.ws_client.connect"
_MODEL = "saidwire-en-2026-10-30"


@pytest.mark.asyncio
async def test_saidwire_success_and_wire_shape(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    provider = SaidwireSTTProvider(stt_settings, model=_MODEL)
    sent: list[Any] = []

    with patch(
        _CONNECT, return_value=fake_connect(load_fixture_events("saidwire"), sent)
    ) as connect:
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.error is None
    assert result.provider == f"saidwire-{_MODEL}"
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is not None
    assert result.vad_events_count == 1
    assert result.complete_transcript == "Hello world, how are you?"
    assert result.word_count == 5

    assert connect.call_args.args[0] == (
        f"wss://us-east.api.saidwire.com/v1/listen?model={_MODEL}"
        "&sample_rate=16000&encoding=linear16&channels=1&interim_results=true"
        "&vad_events=true&punctuate=true&endpointing=false"
    )
    assert connect.call_args.kwargs["additional_headers"] == {
        "Authorization": "Token test-api-key-00000000000000000000"
    }
    control = [json.loads(m) for m in sent if isinstance(m, str)]
    assert control == [{"type": "Finalize"}, {"type": "CloseStream"}]
    assert b"".join(m for m in sent if isinstance(m, bytes)) == audio_pcm_bytes


@pytest.mark.asyncio
async def test_saidwire_stream_without_final_keeps_partial_and_no_final_time(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    events = [e for e in load_fixture_events("saidwire") if not e.get("is_final")]
    provider = SaidwireSTTProvider(stt_settings, model=_MODEL)

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.complete_transcript == "Hello world"
    assert result.audio_to_final_seconds is None


def test_saidwire_construction_guards() -> None:
    with pytest.raises(ValueError, match="saidwire_api_key is required"):
        SaidwireSTTProvider(make_settings(saidwire_api_key=None), model=_MODEL)


def test_saidwire_has_a_provider_env_entry() -> None:
    assert PROVIDER_ENV["saidwire"] == "SAIDWIRE_API_KEY"
