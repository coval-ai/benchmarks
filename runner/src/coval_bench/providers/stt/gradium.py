# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradium real-time STT provider.

Supports models: default
Wire protocol: WebSocket, wss://api.gradium.ai/api/speech/asr
Auth: x-api-key: <key>
Setup: {"type":"setup","model_name":"default","input_format":"pcm_16000",
        "json_config":"{\"language\": \"en\"}"}
Audio: {"type":"audio","audio":"<base64-encoded PCM>"}
Flush: {"type":"flush","flush_id":1}  -- forces buffered results to emit
Close: {"type":"end_of_stream"}

Protocol notes:
- "text" messages carry a word group for the CURRENT segment (no text in "end_text").
- "end_text" signals the segment is finalised; the text comes from the preceding "text".
- A flush must be sent after all audio; wait for the "flushed" ack before
  end_of_stream, otherwise the model's lookahead buffer
  (delay_in_frames ≈ 10 × 80 ms) is discarded.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any, cast

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_URL = "wss://api.gradium.ai/api/speech/asr"


@dataclass
class _GradiumTranscription(Transcription):
    # Every "text" is a final, so final_event can't double as the flush latch.
    flushed: asyncio.Event = field(default_factory=asyncio.Event)


class GradiumSTTProvider(WebSocketSTTProvider):
    """Gradium streaming STT provider."""

    _SAMPLE_RATE = 16000
    _FINAL_WAIT_S = 2.0

    def __init__(self, settings: Settings, model: str = "default") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("gradium_api_key")

    @property
    def name(self) -> str:
        return "gradium"

    def _new_result(self) -> TranscriptionResult:
        return TranscriptionResult(provider=self.name, vad_events_count=0)

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _GradiumTranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
        return _WS_URL

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key.get_secret_value()}

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "setup",
                    "model_name": self._model,
                    "input_format": "pcm_16000",
                    "json_config": json.dumps({"language": "en"}),
                }
            )
        )

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        b64 = base64.b64encode(chunk).decode("utf-8")
        await ws.send(json.dumps({"type": "audio", "audio": b64}))

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        tx = cast(_GradiumTranscription, tx)
        # Flush drains the lookahead buffer; wait for the "flushed" ack so
        # end_of_stream can't cut off pending results.
        frame = json.dumps({"type": "flush", "flush_id": 1})
        await self._send_and_wait(ws, frame, tx, event=tx.flushed)
        await ws.send(json.dumps({"type": "end_of_stream"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_GradiumTranscription, tx)
        result = tx.result
        msg_type: str = msg.get("type", "")

        if msg_type == "step":
            result.vad_events_count = (result.vad_events_count or 0) + 1
            if result.vad_first_detected is None and result.audio_start_time is not None:
                result.vad_first_detected = now - result.audio_start_time
                result.vad_first_event_content = str(msg)
            return False

        if msg_type == "text":
            transcript = str(msg.get("text", "")).strip()
            if transcript:
                tx.partial(transcript, now)
                tx.final(transcript, now)
            return False

        if msg_type == "flushed":
            tx.flushed.set()
            return False

        if msg_type == "end_of_stream":
            return True

        if msg_type == "error":
            tx.fail(str(msg.get("message", msg)))
            self._log.warning("gradium_stt_error", msg=msg)
            return True

        return False
