# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradium real-time STT provider.

Supports models: default
Wire protocol: WebSocket, wss://api.gradium.ai/api/speech/asr
Auth: x-api-key: <key>
Setup: {"type":"setup","model_name":"default","input_format":"pcm"}
Audio: {"type":"audio","audio":"<base64-encoded PCM>"}
Flush: {"type":"flush","flush_id":1}  -- forces buffered results to emit
Close: {"type":"end_of_stream"}

Protocol notes:
- "text" messages carry a word group for the CURRENT segment (no text in "end_text").
- "end_text" signals the segment is finalised; the text comes from the preceding "text".
- A flush must be sent after all audio and waited on before end_of_stream, otherwise
  the model's lookahead buffer (delay_in_frames ≈ 10 × 80 ms) is discarded.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)

_VALID_MODELS = ("default",)
_WS_URL = "wss://api.gradium.ai/api/speech/asr"


class GradiumSTTProvider(STTProvider):
    """Gradium streaming STT provider."""

    def __init__(self, api_key: SecretStr, model: str = "default") -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Gradium STT model {model!r}. Valid: {_VALID_MODELS}")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "gradium"

    @property
    def model(self) -> str:
        return self._model

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
        audio_duration: float | None = None,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name, vad_events_count=0)
        total_start = time.monotonic()

        try:
            headers = {"x-api-key": self._api_key.get_secret_value()}

            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
                await ws.send(
                    json.dumps({"type": "setup", "model_name": self._model, "input_format": "pcm"})
                )

                send_task = asyncio.create_task(
                    self._send_audio(ws, audio_data, sample_rate, result, realtime_resolution)
                )
                recv_task = asyncio.create_task(self._receive(ws, result))
                await asyncio.gather(send_task, recv_task, return_exceptions=True)

        except Exception as exc:
            logger.exception("gradium measure_ttft failed", error=str(exc))
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        sample_rate: int,
        result: TranscriptionResult,
        realtime_resolution: float,
    ) -> None:
        bytes_per_second = sample_rate * 2  # 16-bit mono
        chunk_size = int(bytes_per_second * realtime_resolution)
        result.audio_start_time = time.monotonic()
        try:
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                b64 = base64.b64encode(chunk).decode("utf-8")
                await ws.send(json.dumps({"type": "audio", "audio": b64}))
                await asyncio.sleep(realtime_resolution)
            # Flush forces the model's lookahead buffer to emit pending results.
            # Wait long enough for the server to process remaining frames and
            # send back the final text/end_text messages (~delay_in_frames × 80 ms).
            await ws.send(json.dumps({"type": "flush", "flush_id": 1}))
            await asyncio.sleep(2.0)
            await ws.send(json.dumps({"type": "end_of_stream"}))
        except Exception as exc:
            logger.exception("gradium send error", error=str(exc))
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_parts: list[str] = []
        # "end_text" carries no text — it finalises the segment whose text arrived
        # in the preceding "text" message.  Track it here.
        pending_text: str = ""

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type == "step":
                    result.vad_events_count = (result.vad_events_count or 0) + 1
                    if result.vad_first_detected is None and result.audio_start_time is not None:
                        result.vad_first_detected = now - result.audio_start_time
                        result.vad_first_event_content = str(msg)
                    continue

                if msg_type == "text":
                    transcript = str(msg.get("text", "")).strip()
                    if not transcript:
                        continue
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = now - result.audio_start_time
                        result.first_token_content = (
                            transcript[:30] + "..." if len(transcript) > 30 else transcript
                        )
                    result.partial_transcripts.append(transcript)
                    pending_text = transcript
                    continue

                if msg_type == "end_text":
                    # Commit the text that arrived with the preceding "text" message.
                    if pending_text:
                        final_parts.append(pending_text)
                        if result.audio_start_time is not None:
                            result.audio_to_final_seconds = now - result.audio_start_time
                        pending_text = ""
                    continue

                if msg_type in ("flushed", "ready"):
                    continue

                if msg_type == "end_of_stream":
                    break

                if msg_type == "error":
                    result.error = str(msg.get("message", msg))
                    logger.error("gradium stt error", msg=msg)
                    break

        except Exception as exc:
            logger.exception("gradium receive error", error=str(exc))

        if final_parts:
            result.complete_transcript = " ".join(final_parts).strip() or None
        elif result.partial_transcripts:
            # Fallback: join all word groups received (not just the last one).
            result.complete_transcript = " ".join(result.partial_transcripts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
