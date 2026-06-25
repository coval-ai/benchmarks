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
import contextlib
import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)

_WS_URL = "wss://api.gradium.ai/api/speech/asr"
# Cap on waiting for the "flushed" ack before sending end_of_stream.
_FLUSH_WAIT_S = 2.0


class GradiumSTTProvider(STTProvider):
    """Gradium streaming STT provider."""

    _VALID_MODELS = frozenset({"default"})

    def __init__(self, api_key: SecretStr, model: str = "default") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Gradium STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
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
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name, vad_events_count=0)
        if sample_rate != 16000:
            result.error = f"Gradium requires 16 kHz PCM input; got {sample_rate} Hz"
            return result

        total_start = time.monotonic()

        try:
            headers = {"x-api-key": self._api_key.get_secret_value()}

            final_event = asyncio.Event()
            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
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

                send_task = asyncio.create_task(
                    self._send_audio(
                        ws, audio_data, sample_rate, result, realtime_resolution, final_event
                    )
                )
                recv_task = asyncio.create_task(self._receive(ws, result, final_event))
                outcomes = await asyncio.gather(send_task, recv_task, return_exceptions=True)
                if result.error is None and result.audio_to_final_seconds is None:
                    for outcome in outcomes:
                        if isinstance(outcome, Exception):
                            result.error = str(outcome)
                            break

        except Exception as exc:
            logger.warning(
                "gradium_measure_ttft_failed", provider="gradium", model=self._model, exc_info=exc
            )
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
        final_event: asyncio.Event,
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
            # Flush drains the lookahead buffer; wait for the "flushed" ack so
            # end_of_stream can't cut off pending results.
            await ws.send(json.dumps({"type": "flush", "flush_id": 1}))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(final_event.wait(), timeout=_FLUSH_WAIT_S)
            await ws.send(json.dumps({"type": "end_of_stream"}))
        except Exception as exc:
            logger.warning(
                "gradium_send_error", provider="gradium", model=self._model, exc_info=exc
            )
            raise

    async def _receive(
        self, ws: Any, result: TranscriptionResult, final_event: asyncio.Event
    ) -> None:
        final_parts: list[str] = []

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
                    final_parts.append(transcript)
                    if result.audio_start_time is not None:
                        result.audio_to_final_seconds = now - result.audio_start_time
                    continue

                if msg_type == "flushed":
                    final_event.set()
                    continue

                if msg_type in ("ready", "end_text"):
                    continue

                if msg_type == "end_of_stream":
                    break

                if msg_type == "error":
                    result.error = str(msg.get("message", msg))
                    logger.warning(
                        "gradium_stt_error", provider="gradium", model=self._model, msg=msg
                    )
                    break

        except Exception as exc:
            logger.warning(
                "gradium_receive_error", provider="gradium", model=self._model, exc_info=exc
            )
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        if final_parts:
            result.complete_transcript = " ".join(final_parts).strip() or None
        elif result.partial_transcripts:
            # Fallback: join all word groups received (not just the last one).
            result.complete_transcript = " ".join(result.partial_transcripts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
