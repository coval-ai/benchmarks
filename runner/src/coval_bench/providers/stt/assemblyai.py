# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""AssemblyAI real-time STT provider (v3 streaming API).

Model: universal-streaming
Wire protocol: WebSocket, wss://streaming.assemblyai.com/v3/ws
Auth: Authorization: <key>
Close: {"type": "Terminate"}
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

_VALID_MODELS = ("universal-streaming",)
_WS_URL = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&format_turns=true"


class AssemblyAIProvider(STTProvider):
    """AssemblyAI v3 streaming STT provider."""

    def __init__(self, api_key: SecretStr, model: str = "universal-streaming") -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid AssemblyAI model {model!r}. Valid: {_VALID_MODELS}")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"assemblyai-{self._model}"

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
        if sample_rate != 16000:
            raise ValueError(f"AssemblyAI requires 16 kHz PCM input; got {sample_rate} Hz")

        result = TranscriptionResult(provider=self.name)
        total_start = time.monotonic()

        try:
            headers = {"Authorization": self._api_key.get_secret_value()}
            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
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
            logger.exception("assemblyai measure_ttft failed", error=str(exc))
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
    ) -> None:
        byte_rate = sample_width * sample_rate * channels
        data = audio_data
        first_chunk = True
        try:
            while data:
                chunk_size = int(byte_rate * realtime_resolution)
                chunk, data = data[:chunk_size], data[chunk_size:]
                if first_chunk:
                    result.audio_start_time = time.monotonic()
                    first_chunk = False
                await ws.send(chunk)
                await asyncio.sleep(realtime_resolution)
            await ws.send(json.dumps({"type": "Terminate"}))
        except Exception as exc:
            logger.exception("assemblyai send error", error=str(exc))
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        complete_turns: list[str] = []
        first_end_of_turn_seen = False

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type == "Begin":
                    continue

                transcript = self._extract_transcript(msg)
                if transcript:
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = now - result.audio_start_time
                        result.first_token_content = (
                            transcript[:30] + "..." if len(transcript) > 30 else transcript
                        )
                    result.partial_transcripts.append(transcript)

                if (
                    msg_type == "Turn"
                    and msg.get("end_of_turn")
                    and not first_end_of_turn_seen
                    and transcript
                ):
                    complete_turns.append(transcript)
                    first_end_of_turn_seen = True

        except Exception as exc:
            logger.exception("assemblyai receive error", error=str(exc))

        if complete_turns:
            result.complete_transcript = " ".join(complete_turns).strip() or None
        elif result.partial_transcripts:
            result.complete_transcript = result.partial_transcripts[-1].strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())

    def _extract_transcript(self, msg: dict[str, Any]) -> str:
        if msg.get("type") != "Turn":
            return ""
        transcript = str(msg.get("transcript", "")).strip()
        if transcript:
            return transcript
        words: list[dict[str, Any]] = msg.get("words", [])
        parts = [
            str(w["text"]).strip()
            for w in words
            if isinstance(w, dict) and str(w.get("text", "")).strip()
        ]
        return " ".join(parts)
