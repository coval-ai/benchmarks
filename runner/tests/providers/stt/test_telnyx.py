# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.telnyx (TelnyxSTTProvider).

The WebSocket is replayed with FakeWebSocket; no live network calls are made.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from coval_bench.config import Settings
from coval_bench.providers.stt import STT_PROVIDERS
from coval_bench.providers.stt.telnyx import TelnyxSTTProvider, engine_for
from tests.providers.fakes import FakeWebSocket, fake_connect


def _msg(text: str, *, final: bool) -> dict[str, Any]:
    return {"transcript": text, "is_final": final, "confidence": 0.9 if final else 0.0}


async def _run(
    settings: Settings, audio: bytes, events: list[Any], model: str
) -> tuple[Any, FakeWebSocket, MagicMock]:
    cm = fake_connect(events)
    connect = MagicMock(return_value=cm)
    with patch("coval_bench.providers.stt._streaming.ws_client.connect", connect):
        result = await TelnyxSTTProvider(settings, model=model).measure_ttft(
            audio_data=audio, channels=1, sample_width=2, sample_rate=16000
        )
    return result, cm.ws, connect


def test_telnyx_is_registered() -> None:
    assert STT_PROVIDERS["telnyx"] is TelnyxSTTProvider


@pytest.mark.parametrize(
    ("model", "engine"),
    [
        ("openai/whisper-large-v3-turbo", "Telnyx"),
        ("nvidia/parakeet-v3", "Parakeet"),
        ("deepgram/nova-3", "Deepgram"),
        ("nova-3", "Deepgram"),
        ("xai/grok-stt", "xAI"),
        ("soniox/stt-rt-v4", "Soniox"),
    ],
)
def test_engine_from_vendor_prefix(model: str, engine: str) -> None:
    assert engine_for(model) == engine


def test_unknown_vendor_rejected(stt_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unknown Telnyx STT vendor"):
        TelnyxSTTProvider(stt_settings, model="acme/stt-1")


@pytest.mark.asyncio
async def test_telnyx_whisper_success(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [
        _msg("hel", final=False),
        _msg("hello world", final=True),
        _msg("how are", final=False),
        _msg("how are you", final=True),
    ]
    result, ws, connect = await _run(
        stt_settings, audio_pcm_bytes, events, "openai/whisper-large-v3-turbo"
    )

    assert result.error is None
    assert result.first_token_content == "hel"  # noqa: S105
    assert result.complete_transcript == "hello world how are you"
    assert result.ttft_seconds is not None
    assert result.audio_to_final_seconds is not None

    url = urlparse(connect.call_args.args[0])
    query = parse_qs(url.query)
    assert url.path == "/v2/speech-to-text/transcription"
    assert query["transcription_engine"] == ["Telnyx"]
    assert query["model"] == ["openai/whisper-large-v3-turbo"]
    assert query["input_format"] == ["linear16"]
    assert query["sample_rate"] == ["16000"]
    assert query["interim_results"] == ["true"]
    assert connect.call_args.kwargs["additional_headers"] == {
        "Authorization": "Bearer test-api-key-00000000000000000000"
    }
    # No CloseStream frame for the Telnyx engine; only binary audio goes out.
    assert all(isinstance(m, bytes) for m in ws.sent)
    assert b"".join(ws.sent) == audio_pcm_bytes


@pytest.mark.asyncio
async def test_telnyx_deepgram_sends_close_stream(
    stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    events: list[Any] = [_msg("hello", final=True)]
    result, ws, _ = await _run(stt_settings, audio_pcm_bytes, events, "deepgram/nova-3")

    assert result.error is None
    assert result.complete_transcript == "hello"
    assert json.loads(ws.sent[-1]) == {"type": "CloseStream"}


@pytest.mark.asyncio
async def test_telnyx_pending_partial_kept(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [
        _msg("hello there", final=True),
        _msg("gen", final=False),
        _msg("general kenobi", final=False),
    ]
    result, _, _ = await _run(stt_settings, audio_pcm_bytes, events, "nvidia/parakeet-v3")

    assert result.error is None
    assert result.complete_transcript == "hello there general kenobi"


@pytest.mark.asyncio
async def test_telnyx_error_frame_fails(stt_settings: Settings, audio_pcm_bytes: bytes) -> None:
    events: list[Any] = [
        {
            "errors": [
                {"code": "10009", "title": "Authentication failed", "detail": "Bad key"},
            ]
        }
    ]
    result, _, _ = await _run(stt_settings, audio_pcm_bytes, events, "openai/whisper-tiny")

    assert result.error is not None
    assert "10009" in result.error
    assert "Bad key" in result.error
    assert result.complete_transcript is None


def test_telnyx_missing_key_raises() -> None:
    settings = Settings(telnyx_api_key=None)
    with pytest.raises(ValueError, match="telnyx_api_key is required"):
        TelnyxSTTProvider(settings)
