# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.cloudflare (CloudflareSTTProvider).

The fixture replays the events Workers AI emitted for a live nova-3 stream.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.cloudflare import CloudflareSTTProvider
from coval_bench.registries.provider_keys import PROVIDER_ENV
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events

_CONNECT = "coval_bench.providers.stt.cloudflare.ws_client.connect"
_ACCOUNT = "acct-0123"


def _fake_connect(events: list[Any], captured: dict[str, Any]) -> Any:
    def connect(url: str, additional_headers: dict[str, str], **_: Any) -> Any:
        ws = FakeWebSocket(events)
        captured.update(url=url, headers=additional_headers, ws=ws)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=ws)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    return connect


@pytest.mark.asyncio
async def test_cloudflare_success_and_wire_shape(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    provider = CloudflareSTTProvider(api_key=fake_api_key, model="nova-3", account_id=_ACCOUNT)
    captured: dict[str, Any] = {}

    with patch(_CONNECT, side_effect=_fake_connect(load_fixture_events("cloudflare"), captured)):
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.error is None
    assert result.provider == "cloudflare-nova-3"
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is not None
    assert result.vad_events_count == 1
    assert result.complete_transcript == "Hello world, how are you?"
    assert result.word_count == 5

    assert captured["url"] == (
        f"wss://api.cloudflare.com/client/v4/accounts/{_ACCOUNT}/ai/run/@cf/deepgram/nova-3"
        "?sample_rate=16000&encoding=linear16&channels=1&interim_results=true"
        "&vad_events=true&no_delay=true&punctuate=true&filler_words=true&endpointing=false"
    )
    assert captured["headers"] == {"Authorization": f"Bearer {fake_api_key.get_secret_value()}"}
    control = [json.loads(m) for m in captured["ws"]._sent if isinstance(m, str)]
    assert control == [{"type": "Finalize"}, {"type": "CloseStream"}]
    assert b"".join(m for m in captured["ws"]._sent if isinstance(m, bytes)) == audio_pcm_bytes


@pytest.mark.asyncio
async def test_cloudflare_stream_without_final_keeps_partial_and_no_final_time(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    events = [e for e in load_fixture_events("cloudflare") if not e.get("is_final")]
    provider = CloudflareSTTProvider(api_key=fake_api_key, model="nova-3", account_id=_ACCOUNT)

    with patch(_CONNECT, side_effect=_fake_connect(events, {})):
        result = await provider.measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    assert result.complete_transcript == "Hello world"
    assert result.audio_to_final_seconds is None


def test_cloudflare_construction_guards(fake_api_key: SecretStr) -> None:
    with pytest.raises(ValueError, match="cloudflare_api_key is required"):
        CloudflareSTTProvider(api_key=None, model="nova-3", account_id=_ACCOUNT)
    with pytest.raises(ValueError, match="cloudflare_account_id is required"):
        CloudflareSTTProvider(api_key=fake_api_key, model="nova-3", account_id=None)


def test_cloudflare_has_a_provider_env_entry() -> None:
    assert PROVIDER_ENV["cloudflare"] == "CLOUDFLARE_API_KEY"
