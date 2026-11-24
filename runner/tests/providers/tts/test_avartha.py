# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Avartha WebSocket TTS provider."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts import TTS_PROVIDERS
from coval_bench.providers.tts.avartha import AvarthaTTSProvider
from tests.providers.fakes import FakeWebSocket

from .conftest import make_pcm_bytes

_MODEL = "fishaudio/drama-3"
_VOICE = "cv_0123456789abcdef"


@pytest.mark.asyncio
async def test_avartha_synthesizes_over_stream_input(fake_settings: Settings) -> None:
    ws = FakeWebSocket(
        [
            json.dumps({"audio": base64.b64encode(make_pcm_bytes(240)).decode(), "isFinal": False}),
            json.dumps({"isFinal": True}),
        ]
    )
    captured: dict[str, object] = {}

    def connect_side_effect(url: str, **kwargs: object) -> FakeWebSocket:
        captured["url"] = url
        captured["headers"] = kwargs.get("additional_headers")
        return ws

    provider = AvarthaTTSProvider(fake_settings, model=_MODEL, voice=_VOICE)

    with patch(
        "coval_bench.providers.tts.elevenlabs.ws_client.connect",
        side_effect=connect_side_effect,
    ):
        result = await provider.synthesize("Hello world")

    assert result.error is None, result.error
    assert result.provider == "avartha"
    assert result.model == _MODEL
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    result.audio_path.unlink()

    parts = urlsplit(str(captured["url"]))
    assert parts.netloc == "api.preview.avartha.ai"
    assert parts.path == (
        f"/inference/serverless/elevenlabs/v1/text-to-speech/{_VOICE}/stream-input"
    )
    assert parse_qs(parts.query) == {
        "model_id": [_MODEL],
        "output_format": ["pcm_24000"],
        "auto_mode": ["true"],
        "fresh_take": ["true"],
    }
    assert captured["headers"] == {"xi-api-key": "test-avartha-key"}
    assert [json.loads(frame) for frame in ws.sent] == [
        {"text": " "},
        {"text": "Hello world", "flush": True},
        {"text": ""},
    ]


def test_avartha_skips_fresh_take_outside_drama(fake_settings: Settings) -> None:
    provider = AvarthaTTSProvider(
        fake_settings, model="qwen/qwen3-tts-12hz-1.7b-base", voice="emily"
    )
    url, _, _ = provider._session_frames("Hello world")
    assert "fresh_take" not in parse_qs(urlsplit(url).query)


def test_avartha_registered_and_needs_its_own_key(fake_settings: Settings) -> None:
    assert TTS_PROVIDERS["avartha"] is AvarthaTTSProvider
    settings = fake_settings.model_copy(update={"avartha_api_key": None})
    with pytest.raises(ValueError, match="avartha_api_key"):
        AvarthaTTSProvider(settings, model=_MODEL, voice=_VOICE)
