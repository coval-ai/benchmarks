# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.avartha (AvarthaSTTProvider).

All tests use FakeWebSocket — no live network calls are made. Avartha speaks the
ElevenLabs realtime protocol, so the ElevenLabs fixtures stand in for it.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt import STT_PROVIDERS
from coval_bench.providers.stt.avartha import AvarthaSTTProvider
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings

_CONNECT = "coval_bench.providers.stt._streaming.ws_client.connect"


async def _run(model: str, audio: bytes) -> tuple[Any, dict[str, Any]]:
    settings = make_settings(avartha_api_key=SecretStr("test-key-avartha"), elevenlabs_api_key=None)
    provider = AvarthaSTTProvider(settings, model)
    captured: dict[str, Any] = {}

    def connect_side_effect(
        url: str, additional_headers: dict[str, str] | None = None, **_: Any
    ) -> Any:
        cm = fake_connect(load_fixture_events("elevenlabs", "events-success"))
        captured.update(url=url, headers=dict(additional_headers or {}), ws=cm.ws)
        return cm

    with patch(_CONNECT, side_effect=connect_side_effect):
        result = await provider.measure_ttft(
            audio_data=audio,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )
    return result, captured


@pytest.mark.asyncio
async def test_avartha_success_and_wire_shape(audio_pcm_bytes: bytes) -> None:
    result, captured = await _run("qwen/qwen3-asr-1.7b", audio_pcm_bytes)

    assert result.error is None
    assert result.provider == "avartha-qwen/qwen3-asr-1.7b"
    assert result.complete_transcript == "hello world how are you"

    parts = urlsplit(captured["url"])
    assert parts.netloc == "api.preview.avartha.ai"
    assert parts.path == "/inference/serverless/elevenlabs/v1/speech-to-text/realtime"
    assert parse_qs(parts.query) == {
        "model_id": ["qwen/qwen3-asr-1.7b"],
        "audio_format": ["pcm_16000"],
        "commit_strategy": ["manual"],
        "language_code": ["en"],
    }
    assert captured["headers"] == {"xi-api-key": "test-key-avartha"}

    frames = [json.loads(frame) for frame in captured["ws"].sent]
    assert all(
        set(frame) == {"message_type", "audio_base_64", "sample_rate", "commit"}
        and frame["sample_rate"] == 16000
        for frame in frames
    )
    assert [frame["commit"] for frame in frames] == [False] * (len(frames) - 1) + [True]
    assert frames[-1]["audio_base_64"] == ""


@pytest.mark.asyncio
async def test_avartha_sends_no_language_hint_to_voxtral(audio_pcm_bytes: bytes) -> None:
    _, captured = await _run("avartha/voxtral-mini-4b-realtime-2602-fp8", audio_pcm_bytes)

    assert "language_code" not in parse_qs(urlsplit(captured["url"]).query)


def test_avartha_registered_and_needs_its_own_key() -> None:
    assert STT_PROVIDERS["avartha"] is AvarthaSTTProvider
    with pytest.raises(ValueError, match="avartha_api_key"):
        AvarthaSTTProvider(make_settings(avartha_api_key=None), "qwen/qwen3-asr-1.7b")
