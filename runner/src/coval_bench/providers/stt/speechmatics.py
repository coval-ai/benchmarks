# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Speechmatics real-time STT provider.

Supports models: default, enhanced, broadcast.
Wire protocol: WebSocket, wss://eu2.rt.speechmatics.com/v2
Auth: Authorization: Bearer <key>
Start: {"message": "StartRecognition", ...}
Close: {"message": "EndOfStream", "last_seq_no": N}
"""

from __future__ import annotations

import asyncio
import json
import time  # monotonic clock — wall-clock can step on NTP sync
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)

_VALID_MODELS = ("default", "enhanced", "broadcast")
_WS_URL = "wss://eu2.rt.speechmatics.com/v2"


class SpeechmaticsProvider(STTProvider):
    """Speechmatics real-time STT provider."""

    def __init__(self, api_key: SecretStr, model: str = "default") -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Speechmatics model {model!r}. Valid: {_VALID_MODELS}")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        if self._model == "default":
            return "speechmatics"
        return f"speechmatics-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_start_recognition_config(self, sample_rate: int) -> dict[str, Any]:
        transcription_config: dict[str, Any] = {
            "language": "en",
            "enable_partials": True,
        }
        if self._model == "enhanced":
            transcription_config["operating_point"] = "enhanced"
        elif self._model == "broadcast":
            transcription_config["domain"] = "broadcast"

        return {
            "message": "StartRecognition",
            "transcription_config": transcription_config,
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
            },
        }

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
        audio_duration: float | None = None,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name)
        total_start = time.monotonic()

        try:
            headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
                # Send StartRecognition before streaming audio
                start_config = self._build_start_recognition_config(sample_rate)
                await ws.send(json.dumps(start_config))

                # Wait for RecognitionStarted before sending audio
                await self._wait_for_recognition_started(ws, result)

                result.audio_start_time = time.monotonic()

                send_task = asyncio.create_task(
                    self._send_audio(
                        ws,
                        audio_data,
                        channels,
                        sample_width,
                        sample_rate,
                        result,
                        realtime_resolution,
                    )
                )
                recv_task = asyncio.create_task(self._receive(ws, result))
                await asyncio.gather(send_task, recv_task, return_exceptions=True)

        except Exception as exc:
            logger.exception("speechmatics measure_ttft failed", error=str(exc))
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _wait_for_recognition_started(self, ws: Any, result: TranscriptionResult) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            msg: dict[str, Any] = json.loads(raw)
            if msg.get("message") == "RecognitionStarted":
                return
            if msg.get("message") == "Error":
                raise RuntimeError(f"Speechmatics error: {msg.get('reason', 'Unknown error')}")

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        result: TranscriptionResult,
        realtime_resolution: float,
    ) -> None:
        byte_rate = sample_width * sample_rate * channels
        data = audio_data
        seq_no = 0
        try:
            while data:
                chunk_size = int(byte_rate * realtime_resolution)
                chunk, data = data[:chunk_size], data[chunk_size:]
                await ws.send(chunk)
                seq_no += 1
                await asyncio.sleep(realtime_resolution)
            await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": seq_no}))
        except Exception as exc:
            logger.exception("speechmatics send error", error=str(exc))
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_transcripts: list[str] = []
        last_final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("message", "")

                if msg_type == "AudioAdded":
                    continue
                if msg_type == "EndOfTranscript":
                    break
                if msg_type == "Error":
                    reason: str = msg.get("reason", "Unknown error")
                    if "EndOfStream" not in reason and "schema" not in reason.lower():
                        result.error = reason
                    continue
                if msg_type in ("Warning", "Info"):
                    continue

                transcript = self._extract_transcript(msg)
                if transcript:
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = now - result.audio_start_time
                        result.first_token_content = transcript
                    result.partial_transcripts.append(transcript)

                if msg_type == "AddTranscript":
                    final_transcripts.append(transcript)
                    last_final_time = now

        except Exception as exc:
            logger.exception("speechmatics receive error", error=str(exc))

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        if final_transcripts:
            result.complete_transcript = " ".join(final_transcripts).strip() or None
            if result.complete_transcript:
                result.transcript_length = len(result.complete_transcript)
                result.word_count = len(result.complete_transcript.split())

    def _extract_transcript(self, msg: dict[str, Any]) -> str:
        msg_type: str = msg.get("message", "")
        if msg_type not in ("AddTranscript", "AddPartialTranscript"):
            return ""
        results: list[dict[str, Any]] = msg.get("results", [])
        parts: list[str] = []
        for r in results:
            alternatives: list[dict[str, Any]] = r.get("alternatives", [])
            if alternatives:
                content = alternatives[0].get("content", "")
                if content:
                    parts.append(content)
        return " ".join(parts).strip()
