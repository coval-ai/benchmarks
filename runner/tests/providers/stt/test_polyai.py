# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.polyai (PolyAISTTProvider).

All tests use FakeWebSocket — no live network calls are made. Owl speaks the
OpenAI Realtime transcription protocol, so the OpenAI fixtures stand in for it.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt import STT_PROVIDERS
from coval_bench.providers.stt.polyai import PolyAISTTProvider
from tests.providers.fakes import fake_connect
from tests.providers.stt.conftest import load_fixture_events, make_settings

_CONNECT = "coval_bench.providers.stt._streaming.ws_client.connect"


@pytest.mark.asyncio
async def test_polyai_success_and_wire_shape(audio_pcm_bytes: bytes) -> None:
    settings = make_settings(polyai_owl_api_key=SecretStr("test-key-polyai"), openai_api_key=None)
    provider = PolyAISTTProvider(settings)
    captured: dict[str, Any] = {}

    def connect_side_effect(
        url: str, additional_headers: dict[str, str] | None = None, **_: Any
    ) -> Any:
        cm = fake_connect(load_fixture_events("openai", "events-success"))
        captured.update(url=url, headers=dict(additional_headers or {}), ws=cm.ws)
        return cm

    with patch(_CONNECT, side_effect=connect_side_effect):
        result = await provider.measure_ttft(
            audio_data=audio_pcm_bytes,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )

    assert result.error is None
    assert result.provider == "polyai-dialog-owl-1.0"
    assert result.complete_transcript == "hello world how are you"

    assert captured["url"] == "wss://owl.polyai-tools.com/v1/realtime?intent=transcription"
    assert captured["headers"] == {"Authorization": "Bearer test-key-polyai"}
    session_update = json.loads(captured["ws"].sent[0])
    assert session_update["type"] == "session.update"
    assert session_update["session"]["audio"]["input"]["transcription"]["model"] == "dialog-owl-1.0"


def test_polyai_registered_and_needs_its_own_key() -> None:
    assert STT_PROVIDERS["polyai"] is PolyAISTTProvider
    with pytest.raises(ValueError, match="polyai_owl_api_key"):
        PolyAISTTProvider(make_settings(polyai_owl_api_key=None))
