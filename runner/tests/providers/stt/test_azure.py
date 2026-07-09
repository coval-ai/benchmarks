# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.azure (AzureSTTProvider).

All tests use FakeWebSocket — no live network calls are made. Server frames are
built with the Azure header-block framing (``Path:...\\r\\n...\\r\\n\\r\\n<json>``).
"""

from __future__ import annotations

import json
import struct
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from coval_bench.providers.stt.azure import (
    AzureSTTProvider,
    _audio_message,
    _parse_message,
    _wav_header,
)
from tests.providers.stt.conftest import FakeWebSocket

REGION = "eastus"


def make_provider(model: str = "default") -> AzureSTTProvider:
    return AzureSTTProvider(api_key=SecretStr("test-key-azure"), model=model, region=REGION)


def _frame(path: str, body: dict[str, Any] | None = None) -> str:
    """Frame a server text message the way Azure sends them."""
    header = f"X-RequestId:req-1\r\nContent-Type:application/json\r\nPath:{path}\r\n"
    return header + "\r\n" + (json.dumps(body) if body is not None else "")


def _phrase(text: str, status: str = "Success") -> dict[str, Any]:
    return {
        "RecognitionStatus": status,
        "Offset": 0,
        "Duration": 1000,
        "NBest": [{"Confidence": 0.95, "Lexical": text.lower(), "Display": text}],
    }


def _fake_connect(events: list[Any], on_send: Any = None) -> Any:
    ws = FakeWebSocket(events, on_send=on_send)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


async def _run(
    provider: AzureSTTProvider, events: list[Any], audio: bytes, on_send: Any = None
) -> Any:
    with patch(
        "coval_bench.providers.stt.azure.ws_client.connect",
        return_value=_fake_connect(events, on_send=on_send),
    ):
        return await provider.measure_ttft(
            audio_data=audio,
            channels=1,
            sample_width=2,
            sample_rate=16000,
            realtime_resolution=0.5,
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_azure_success(audio_pcm_bytes: bytes) -> None:
    events = [
        _frame("turn.start", {"context": {"serviceTag": "x"}}),
        _frame("speech.startDetected", {"Offset": 100}),
        _frame("speech.hypothesis", {"Text": "hello", "Offset": 0, "Duration": 100}),
        _frame("speech.phrase", _phrase("Hello world")),
        _frame("speech.endDetected", {"Offset": 900}),
        _frame("turn.end"),
    ]
    result = await _run(make_provider(), events, audio_pcm_bytes)

    assert result.error is None
    assert result.ttft_seconds is not None and result.ttft_seconds >= 0
    assert result.first_token_content is not None
    assert "hello" in result.first_token_content.lower()
    assert result.complete_transcript == "Hello world"
    assert result.word_count == 2
    assert result.audio_to_final_seconds is not None
    assert result.vad_first_detected is not None
    assert result.vad_events_count == 1


@pytest.mark.asyncio
async def test_azure_multi_phrase_concatenates(audio_pcm_bytes: bytes) -> None:
    """Conversation mode emits multiple finals across pauses; all are joined."""
    events = [
        _frame("turn.start"),
        _frame("speech.phrase", _phrase("Hello world")),
        _frame("speech.phrase", _phrase("how are you")),
        _frame("turn.end"),
    ]
    result = await _run(make_provider(), events, audio_pcm_bytes)

    assert result.complete_transcript == "Hello world how are you"
    assert result.audio_to_final_seconds is not None


@pytest.mark.asyncio
async def test_azure_simple_format_display_text_fallback(audio_pcm_bytes: bytes) -> None:
    """A speech.phrase without NBest falls back to the top-level DisplayText."""
    events = [
        _frame("turn.start"),
        _frame(
            "speech.phrase",
            {"RecognitionStatus": "Success", "DisplayText": "plain text", "Offset": 0},
        ),
        _frame("turn.end"),
    ]
    result = await _run(make_provider(), events, audio_pcm_bytes)

    assert result.complete_transcript == "plain text"


@pytest.mark.asyncio
async def test_azure_non_success_phrase_ignored(audio_pcm_bytes: bytes) -> None:
    """NoMatch / silence-timeout phrases carry no text and must not be scored."""
    events = [
        _frame("turn.start"),
        _frame("speech.phrase", {"RecognitionStatus": "NoMatch"}),
        _frame("turn.end"),
    ]
    result = await _run(make_provider(), events, audio_pcm_bytes)

    assert result.complete_transcript is None
    assert result.audio_to_final_seconds is None


@pytest.mark.asyncio
async def test_azure_hypothesis_only_leaves_transcript_none(audio_pcm_bytes: bytes) -> None:
    """An unfinalized hypothesis sets ttft/partials but never the scored transcript."""
    events = [
        _frame("turn.start"),
        _frame("speech.hypothesis", {"Text": "hello"}),
        _frame("speech.hypothesis", {"Text": "hello world"}),
        _frame("turn.end"),
    ]
    result = await _run(make_provider(), events, audio_pcm_bytes)

    assert result.complete_transcript is None
    assert result.ttft_seconds is not None
    assert result.partial_transcripts == ["hello", "hello world"]


@pytest.mark.asyncio
async def test_azure_empty_stream(audio_pcm_bytes: bytes) -> None:
    result = await _run(make_provider(), [], audio_pcm_bytes)
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_azure_sends_speech_config_then_ends_with_empty_audio(
    audio_pcm_bytes: bytes,
) -> None:
    sent: list[Any] = []
    events = [_frame("turn.start"), _frame("speech.phrase", _phrase("hi")), _frame("turn.end")]
    await _run(make_provider(), events, audio_pcm_bytes, on_send=sent.append)

    # First message is the speech.config text frame.
    assert isinstance(sent[0], str)
    assert "Path:speech.config" in sent[0]

    binary = [m for m in sent if isinstance(m, (bytes, bytearray))]
    assert binary, "expected binary audio frames"
    # The final audio frame is header-only (empty payload) — the end-of-stream signal.
    last = binary[-1]
    header_len = struct.unpack(">H", last[:2])[0]
    assert len(last) == 2 + header_len
    assert b"Path:audio" in last[2 : 2 + header_len]


@pytest.mark.asyncio
async def test_azure_url_uses_region_and_conversation_mode(audio_pcm_bytes: bytes) -> None:
    events = [_frame("turn.end")]
    with patch(
        "coval_bench.providers.stt.azure.ws_client.connect",
        return_value=_fake_connect(events),
    ) as mock_connect:
        await make_provider().measure_ttft(audio_pcm_bytes, 1, 2, 16000, 0.5)

    url = mock_connect.call_args.args[0]
    assert url.startswith(f"wss://{REGION}.stt.speech.microsoft.com/")
    assert "/recognition/conversation/" in url
    assert "language=en-US" in url
    assert "format=detailed" in url


def test_audio_message_round_trips() -> None:
    payload = b"\x01\x02\x03\x04"
    frame = _audio_message("req-abc", payload)
    header_len = struct.unpack(">H", frame[:2])[0]
    header = frame[2 : 2 + header_len].decode("ascii")
    assert "Path:audio" in header
    assert "X-RequestId:req-abc" in header
    assert frame[2 + header_len :] == payload


def test_parse_message_extracts_path_and_body() -> None:
    path, body = _parse_message(_frame("speech.phrase", {"RecognitionStatus": "Success"}))
    assert path == "speech.phrase"
    assert body["RecognitionStatus"] == "Success"

    path, body = _parse_message(_frame("turn.end"))
    assert path == "turn.end"
    assert body == {}


def test_wav_header_is_canonical_pcm() -> None:
    header = _wav_header(16000, 1, 2)
    assert len(header) == 44
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert struct.unpack("<I", header[24:28])[0] == 16000  # sample rate
    assert struct.unpack("<H", header[22:24])[0] == 1  # channels
    assert struct.unpack("<H", header[34:36])[0] == 16  # bits per sample


# ---------------------------------------------------------------------------
# Provider identity + validation
# ---------------------------------------------------------------------------


def test_provider_name() -> None:
    assert make_provider().name == "azure"


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Azure model"):
        AzureSTTProvider(api_key=SecretStr("k"), model="bad-model", region=REGION)


def test_missing_region_raises() -> None:
    with pytest.raises(ValueError, match="requires region"):
        AzureSTTProvider(api_key=SecretStr("k"))


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="requires api_key"):
        AzureSTTProvider(api_key=None, region=REGION)  # type: ignore[arg-type]
