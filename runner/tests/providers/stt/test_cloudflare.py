# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.cloudflare (CloudflareSTTProvider).

The fixture replays the events Workers AI emitted for a live nova-3 stream.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from coval_bench.config import Settings
from coval_bench.providers.stt import cloudflare
from coval_bench.providers.stt.cloudflare import CloudflareSTTProvider
from coval_bench.registries.provider_keys import PROVIDER_ENV
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings

_CONNECT = "coval_bench.providers.stt._streaming.ws_client.connect"
_ACCOUNT = "acct-0123"


@pytest.fixture(autouse=True)
def _no_final_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake replays every event before Finalize, so the wait could only time out."""
    monkeypatch.setattr(cloudflare.CloudflareSTTProvider, "_FINAL_WAIT_S", 0.0)


@pytest.mark.asyncio
async def test_cloudflare_success_and_wire_shape(audio_pcm_bytes: bytes) -> None:
    provider = CloudflareSTTProvider(make_settings(cloudflare_account_id=_ACCOUNT), model="nova-3")
    sent: list[Any] = []

    with patch(
        _CONNECT, return_value=fake_connect(load_fixture_events("cloudflare"), sent)
    ) as connect:
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.error is None
    assert result.provider == "cloudflare-nova-3"
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is not None
    assert result.vad_events_count == 1
    assert result.complete_transcript == "Hello world, how are you?"
    assert result.word_count == 5

    assert connect.call_args.args[0] == (
        f"wss://api.cloudflare.com/client/v4/accounts/{_ACCOUNT}/ai/run/@cf/deepgram/nova-3"
        "?sample_rate=16000&encoding=linear16&channels=1&interim_results=true"
        "&vad_events=true&no_delay=true&punctuate=true&filler_words=true&endpointing=false"
    )
    assert connect.call_args.kwargs["additional_headers"] == {
        "Authorization": "Bearer test-api-key-00000000000000000000"
    }
    control = [json.loads(m) for m in sent if isinstance(m, str)]
    assert control == [{"type": "Finalize"}, {"type": "CloseStream"}]
    assert b"".join(m for m in sent if isinstance(m, bytes)) == audio_pcm_bytes


@pytest.mark.asyncio
async def test_cloudflare_stream_without_final_keeps_partial_and_no_final_time(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    events = [e for e in load_fixture_events("cloudflare") if not e.get("is_final")]
    provider = CloudflareSTTProvider(stt_settings, model="nova-3")

    with patch(_CONNECT, return_value=fake_connect(events)):
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.complete_transcript == "Hello world"
    assert result.audio_to_final_seconds is None


def test_cloudflare_construction_guards() -> None:
    with pytest.raises(ValueError, match="cloudflare_api_key is required"):
        CloudflareSTTProvider(make_settings(cloudflare_api_key=None), model="nova-3")
    with pytest.raises(ValueError, match="cloudflare_account_id is required"):
        CloudflareSTTProvider(make_settings(cloudflare_account_id=None), model="nova-3")


def test_cloudflare_has_a_provider_env_entry() -> None:
    assert PROVIDER_ENV["cloudflare"] == "CLOUDFLARE_API_KEY"
