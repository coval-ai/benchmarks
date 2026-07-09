# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Microsoft Azure AI Speech real-time STT provider.

Drives the Speech WebSocket protocol directly, not the Speech SDK: the SDK adds
seconds of event-delivery overhead that would inflate latency vs. other providers.

Endpoint: wss://{region}.stt.speech.microsoft.com/speech/recognition/conversation
/cognitiveservices/v1. Auth: Ocp-Apim-Subscription-Key header. Client text frames
are a header block + JSON body; audio frames are a 2-byte big-endian header length,
header, then payload; an empty payload ends the stream. Conversation mode is
continuous, so the transcript concatenates every speech.phrase final until turn.end.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time  # monotonic clock — wall-clock can step on NTP sync
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks
from coval_bench.providers.stt._transcript_utils import (
    add_partial_transcript,
    set_first_token,
)

logger = structlog.get_logger(__name__)

# conversation mode keeps recognizing across pauses; interactive stops at the first.
_WS_PATH = "/speech/recognition/conversation/cognitiveservices/v1"
_LANGUAGE = "en-US"

_SPEECH_CONFIG = {
    "context": {
        "system": {"name": "coval-bench", "version": "1.0"},
        "os": {"platform": "Linux", "name": "Linux", "version": "1.0"},
    }
}


class AzureSTTProvider(STTProvider):
    """Azure AI Speech real-time streaming STT provider (raw WebSocket)."""

    _VALID_MODELS = frozenset({"default"})

    def __init__(
        self,
        api_key: SecretStr,
        model: str = "default",
        region: str | None = None,
    ) -> None:
        if not self._model_supported(model):
            raise ValueError(f"Invalid Azure model {model!r}. Valid: {sorted(self._VALID_MODELS)}")
        if region is None:
            raise ValueError("AzureSTTProvider requires region (set Settings.azure_region)")
        if api_key is None:
            raise ValueError("AzureSTTProvider requires api_key (set Settings.azure_api_key)")
        self._api_key = api_key
        self._model = model
        self._region = region

    @property
    def name(self) -> str:
        return "azure"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self) -> str:
        return (
            f"wss://{self._region}.stt.speech.microsoft.com{_WS_PATH}"
            f"?language={_LANGUAGE}&format=detailed"
        )

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name, vad_events_count=0)
        total_start = time.monotonic()

        try:
            url = self._build_websocket_url()
            request_id = uuid.uuid4().hex
            headers = {
                "Ocp-Apim-Subscription-Key": self._api_key.get_secret_value(),
                "X-ConnectionId": uuid.uuid4().hex,
            }
            async with ws_client.connect(url, additional_headers=headers) as ws:
                await ws.send(_text_message("speech.config", json.dumps(_SPEECH_CONFIG)))
                send_task = asyncio.create_task(
                    self._send_audio(
                        ws,
                        audio_data,
                        channels,
                        sample_width,
                        sample_rate,
                        result,
                        realtime_resolution,
                        request_id,
                    )
                )
                recv_task = asyncio.create_task(self._receive(ws, result))
                tasks = (send_task, recv_task)
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                if any(not task.cancelled() and task.exception() is not None for task in done):
                    for task in pending:
                        task.cancel()
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                if result.error is None and result.audio_to_final_seconds is None:
                    for outcome in outcomes:
                        if isinstance(outcome, Exception):
                            result.error = str(outcome)
                            break

        except Exception as exc:
            logger.warning(
                "azure_measure_ttft_failed", provider="azure", model=self._model, exc_info=exc
            )
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        result: TranscriptionResult,
        realtime_resolution: float,
        request_id: str,
    ) -> None:
        # Payloads are concatenated server-side, so the RIFF header prefixes the PCM.
        stream = _wav_header(sample_rate, channels, sample_width) + audio_data
        byte_rate = sample_width * sample_rate * channels
        chunk_size = int(byte_rate * realtime_resolution)
        try:
            async for chunk, start in paced_chunks(stream, chunk_size, byte_rate):
                if result.audio_start_time is None:
                    result.audio_start_time = start
                await ws.send(_audio_message(request_id, chunk))
            await ws.send(_audio_message(request_id, b""))  # empty payload = end of stream
        except Exception as exc:
            logger.warning("azure_send_error", provider="azure", model=self._model, exc_info=exc)
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_segments: list[str] = []
        last_final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                path, body = _parse_message(raw)
                now = time.monotonic()

                if path == "speech.startdetected":
                    if result.audio_start_time is not None and result.vad_first_detected is None:
                        result.vad_first_detected = now - result.audio_start_time
                        result.vad_first_event_content = str(body)
                    result.vad_events_count = (result.vad_events_count or 0) + 1
                    continue

                if path == "speech.hypothesis":
                    transcript = str(body.get("Text", "")).strip()
                    if transcript:
                        set_first_token(result, transcript, now=now)
                        add_partial_transcript(result, transcript)
                    continue

                if path == "speech.phrase":
                    if body.get("RecognitionStatus") != "Success":
                        continue
                    transcript = _phrase_text(body)
                    if not transcript:
                        continue
                    set_first_token(result, transcript, now=now)
                    add_partial_transcript(result, transcript)
                    final_segments.append(transcript)
                    last_final_time = now
                    continue

                if path == "turn.end":
                    break

        except Exception as exc:
            logger.warning("azure_receive_error", provider="azure", model=self._model, exc_info=exc)
            if result.error is None and last_final_time is None:
                result.error = str(exc)

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        # Only finalized phrases are scored; unfinalized hypotheses stay out.
        if final_segments:
            result.complete_transcript = " ".join(final_segments).strip() or None
        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _text_message(path: str, body: str) -> str:
    header = f"Path:{path}\r\nX-Timestamp:{_timestamp()}\r\nContent-Type:application/json\r\n"
    return f"{header}\r\n{body}"


def _audio_message(request_id: str, payload: bytes) -> bytes:
    header = (
        f"Path:audio\r\nX-RequestId:{request_id}\r\n"
        f"X-Timestamp:{_timestamp()}\r\nContent-Type:audio/x-wav\r\n"
    ).encode("ascii")
    return struct.pack(">H", len(header)) + header + payload


def _parse_message(raw: str) -> tuple[str, dict[str, Any]]:
    """Split a server text frame into (lowercased Path, JSON body)."""
    head, _, body = raw.partition("\r\n\r\n")
    path = ""
    for line in head.split("\r\n"):
        name, _, value = line.partition(":")
        if name.strip().lower() == "path":
            path = value.strip().lower()
            break
    try:
        data: dict[str, Any] = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError:
        data = {}
    return path, data


def _phrase_text(body: dict[str, Any]) -> str:
    """Best display text from a speech.phrase body (detailed NBest or simple)."""
    nbest = body.get("NBest")
    if isinstance(nbest, list) and nbest:
        first = nbest[0]
        if isinstance(first, dict):
            return str(first.get("Display", "") or first.get("Lexical", "")).strip()
    return str(body.get("DisplayText", "")).strip()


def _wav_header(sample_rate: int, channels: int, sample_width: int) -> bytes:
    """44-byte PCM WAV header. Sizes are 0: the service streams and ignores them."""
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    return (
        b"RIFF"
        + struct.pack("<I", 0)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)  # PCM
        + struct.pack("<H", channels)
        + struct.pack("<I", sample_rate)
        + struct.pack("<I", byte_rate)
        + struct.pack("<H", block_align)
        + struct.pack("<H", sample_width * 8)
        + b"data"
        + struct.pack("<I", 0)
    )
