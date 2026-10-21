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

import json
import struct
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

# conversation mode keeps recognizing across pauses; interactive stops at the first.
_WS_PATH = "/speech/recognition/conversation/cognitiveservices/v1"
_LANGUAGE = "en-US"

_SPEECH_CONFIG = {
    "context": {
        "system": {"name": "coval-bench", "version": "1.0"},
        "os": {"platform": "Linux", "name": "Linux", "version": "1.0"},
    }
}


@dataclass
class _AzureTranscription(Transcription):
    # Stamped on every audio frame of one turn.
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class AzureSTTProvider(WebSocketSTTProvider):
    """Azure AI Speech real-time streaming STT provider (raw WebSocket)."""

    def __init__(self, settings: Settings, model: str = "default") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("azure_api_key")
        self._region = self._require_setting("azure_region")

    @property
    def name(self) -> str:
        return "azure"

    def _new_result(self) -> TranscriptionResult:
        return TranscriptionResult(provider=self.name, vad_events_count=0)

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _AzureTranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
        return (
            f"wss://{self._region}.stt.speech.microsoft.com{_WS_PATH}"
            f"?language={_LANGUAGE}&format=detailed"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Ocp-Apim-Subscription-Key": self._api_key.get_secret_value(),
            "X-ConnectionId": uuid.uuid4().hex,
        }

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(_text_message("speech.config", json.dumps(_SPEECH_CONFIG)))

    # Wholesale: every audio frame carries the per-call request id, which the
    # chunk hook cannot see.
    async def _send(
        self,
        ws: Any,
        audio_data: bytes,
        sample_rate: int,
        realtime_resolution: float,
        tx: Transcription,
    ) -> None:
        tx = cast(_AzureTranscription, tx)
        # Payloads are concatenated server-side, so the RIFF header prefixes the PCM.
        stream = _wav_header(sample_rate, 1, 2) + audio_data
        byte_rate = sample_rate * 2
        chunk_size = max(2, int(byte_rate * realtime_resolution))
        try:
            async for chunk, start in paced_chunks(stream, chunk_size, byte_rate):
                tx.result.audio_start_time = start
                await ws.send(_audio_message(tx.request_id, chunk))
            await self._end_audio(ws, tx)
        except Exception as exc:
            self._log.warning("stt_send_error", exc_info=exc)
            raise

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        tx = cast(_AzureTranscription, tx)
        await ws.send(_audio_message(tx.request_id, b""))  # empty payload = end of stream

    def _decode(self, raw: str | bytes) -> Any:
        return _parse_message(raw if isinstance(raw, str) else raw.decode())

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        path, body = msg
        result = tx.result

        if path == "speech.startdetected":
            if result.audio_start_time is not None and result.vad_first_detected is None:
                result.vad_first_detected = now - result.audio_start_time
                result.vad_first_event_content = str(body)
            result.vad_events_count = (result.vad_events_count or 0) + 1
            return False

        if path == "speech.hypothesis":
            tx.partial(str(body.get("Text", "")), now)
            return False

        if path == "speech.phrase":
            if body.get("RecognitionStatus") != "Success":
                return False
            transcript = _phrase_text(body)
            if transcript:
                tx.partial(transcript, now)
                tx.final(transcript, now)
            return False

        return bool(path == "turn.end")


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
