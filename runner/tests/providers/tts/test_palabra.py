# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Palabra real-time TTS provider."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.providers.tts.palabra import PalabraTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes

_MODEL = "palabra-tts-v1"
_FEMALE = "36b6daf6-4792-4106-a0ad-0069f582e7f8"
_MALE = "f1d62423-085e-4694-91cd-4d04085a2fc5"


@pytest.fixture()
def palabra_settings() -> Settings:
    return Settings(
        palabra_api_key=SecretStr("test-palabra-key"),
    )


def _audio_events(pcm: bytes) -> list[str | bytes]:
    return [
        json.dumps(
            {
                "message_type": "audio_chunk",
                "data": {"audio": base64.b64encode(pcm).decode("ascii"), "last_chunk": True},
            }
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", ["default_low", _FEMALE, _MALE])
async def test_palabra_tts_sends_voice(palabra_settings: Settings, voice: str) -> None:
    ws = FakeWebSocket(_audio_events(make_pcm_bytes(240)))
    provider = PalabraTTSProvider(palabra_settings, model=_MODEL, voice=voice)

    with patch("coval_bench.providers.tts.palabra.ws_client.connect", return_value=ws):
        result = await provider.synthesize("Hello from Palabra")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.voice == voice
    init = json.loads(ws.sent[0])
    assert init["voice_options"] == {"voice_id": voice}
    assert result.audio_path is not None
    result.audio_path.unlink()


def test_palabra_tts_rejects_unknown_voice(palabra_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Invalid Palabra TTS voice"):
        PalabraTTSProvider(palabra_settings, model=_MODEL, voice="nonexistent")
