# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.together (TogetherSTTProvider).

The WebSocket is replayed with FakeWebSocket — no live network calls. Deltas
are incremental fragments; each ``completed`` event carries a full utterance
transcript, so the complete transcript is the in-order join of the finals.
"""

from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.metrics.wer import compute_wer
from coval_bench.providers.stt.together import TogetherSTTProvider
from tests.providers.stt.conftest import FakeWebSocket, load_fixture_events

_CHUNK_BYTES = int(16000 * 2 * 0.1)


def _fake_ws_connect(ws: FakeWebSocket) -> Any:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _frames(sent: list[Any]) -> list[dict[str, Any]]:
    return [json.loads(f) for f in sent if isinstance(f, str)]


async def _run(provider: TogetherSTTProvider, ws: FakeWebSocket, audio: bytes) -> Any:
    with patch(
        "coval_bench.providers.stt.together.ws_client.connect",
        return_value=_fake_ws_connect(ws),
    ):
        return await provider.measure_ttft(
            audio_data=audio, channels=1, sample_width=2, sample_rate=16000
        )


@pytest.mark.asyncio
async def test_together_success(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    ws = FakeWebSocket(load_fixture_events("together"), server_closes=False)
    provider = TogetherSTTProvider(api_key=fake_api_key)

    result = await _run(provider, ws, audio_pcm_bytes)

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.ttft_seconds >= 0
    assert result.first_token_content == "hello"  # noqa: S105
    assert result.complete_transcript == "hello world how are you"
    assert result.word_count == 5
    assert result.audio_to_final_seconds is not None
    wer = compute_wer("hello world how are you", result.complete_transcript)
    assert wer.wer_percentage == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_together_commit_is_last_frame(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    sent: list[Any] = []
    ws = FakeWebSocket(load_fixture_events("together"), on_send=sent.append, server_closes=False)
    provider = TogetherSTTProvider(api_key=fake_api_key)

    await _run(provider, ws, audio_pcm_bytes)

    frames = _frames(sent)
    assert frames[-1]["type"] == "input_audio_buffer.commit"
    appends = [f for f in frames if f["type"] == "input_audio_buffer.append"]
    assert len(appends) == math.ceil(len(audio_pcm_bytes) / _CHUNK_BYTES)


@pytest.mark.asyncio
async def test_together_nemotron_pads_tail_silence(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """Nemotron models get trailing silence appended before the commit."""
    sent: list[Any] = []
    ws = FakeWebSocket(load_fixture_events("together"), on_send=sent.append, server_closes=False)
    provider = TogetherSTTProvider(api_key=fake_api_key, model="nemotron-3.5-asr-streaming-0.6b")

    await _run(provider, ws, audio_pcm_bytes)

    frames = _frames(sent)
    assert frames[-1]["type"] == "input_audio_buffer.commit"
    appends = [f for f in frames if f["type"] == "input_audio_buffer.append"]
    assert len(appends) > math.ceil(len(audio_pcm_bytes) / _CHUNK_BYTES)


@pytest.mark.asyncio
async def test_together_deltas_only_falls_back_to_concatenation(
    fake_api_key: SecretStr, audio_pcm_bytes: bytes
) -> None:
    """With no final, the concatenated deltas are salvaged as the transcript."""
    events: list[Any] = [
        {"type": "conversation.item.input_audio_transcription.delta", "delta": "hello"},
        {"type": "conversation.item.input_audio_transcription.delta", "delta": " world"},
    ]
    ws = FakeWebSocket(events, server_closes=False)
    provider = TogetherSTTProvider(api_key=fake_api_key)

    result = await _run(provider, ws, audio_pcm_bytes)

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.complete_transcript == "hello world"
    assert result.audio_to_final_seconds is None


def test_provider_name() -> None:
    provider = TogetherSTTProvider(api_key=SecretStr("k"), model="parakeet-tdt-0.6b-v3")
    assert provider.name == "together-parakeet-tdt-0.6b-v3"


def test_provider_model() -> None:
    provider = TogetherSTTProvider(api_key=SecretStr("k"))
    assert provider.model == "whisper-large-v3"


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Together STT model"):
        TogetherSTTProvider(api_key=SecretStr("k"), model="whisper-large-v2")


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="together_api_key is required"):
        TogetherSTTProvider(api_key=None)


@pytest.mark.asyncio
async def test_together_error_event(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    ws = FakeWebSocket(load_fixture_events("together", "events-error"), server_closes=False)
    provider = TogetherSTTProvider(api_key=fake_api_key)

    result = await _run(provider, ws, audio_pcm_bytes)

    assert result.error is not None
    assert "Invalid API key" in result.error
    assert result.complete_transcript is None


@pytest.mark.asyncio
async def test_together_unsupported_audio_format(fake_api_key: SecretStr) -> None:
    provider = TogetherSTTProvider(api_key=fake_api_key)
    result = await provider.measure_ttft(
        audio_data=b"\x00\x00", channels=2, sample_width=2, sample_rate=44100
    )
    assert result.error is not None
    assert "16 kHz mono" in result.error


@pytest.mark.asyncio
async def test_together_empty_stream(fake_api_key: SecretStr, audio_pcm_bytes: bytes) -> None:
    ws = FakeWebSocket([], server_closes=False)
    provider = TogetherSTTProvider(api_key=fake_api_key)

    result = await _run(provider, ws, audio_pcm_bytes)

    assert result.complete_transcript is None
    assert result.ttft_seconds is None
