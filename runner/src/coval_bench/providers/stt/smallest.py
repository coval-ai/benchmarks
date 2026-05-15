# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Smallest AI Pulse real-time STT provider.

Supports models: default (pulse)
Wire protocol: WebSocket, wss://api.smallest.ai/waves/v1/pulse/get_text
Auth: Authorization: Bearer <key>
Query params: language=en, encoding=linear16, sample_rate=<hz>
Audio: raw binary PCM chunks streamed at realtime pace
Close: {"type": "close_stream"}  — server drains buffer, then sends is_last=true
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)

_VALID_MODELS = ("pulse",)
_WS_BASE_URL = "wss://api.smallest.ai/waves/v1/pulse/get_text"


class SmallestSTTProvider(STTProvider):
    """Smallest AI Pulse streaming STT provider."""

    def __init__(self, api_key: SecretStr, model: str = "pulse") -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Smallest STT model {model!r}. Valid: {_VALID_MODELS}")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "smallest"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self, sample_rate: int) -> str:
        return (
            f"{_WS_BASE_URL}"
            f"?language=en"
            f"&encoding=linear16"
            f"&sample_rate={sample_rate}"
        )

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
            url = self._build_websocket_url(sample_rate)
            headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

            async with ws_client.connect(url, additional_headers=headers) as ws:
                send_task = asyncio.create_task(
                    self._send_audio(ws, audio_data, sample_rate, result, realtime_resolution)
                )
                recv_task = asyncio.create_task(self._receive(ws, result))
                await asyncio.gather(send_task, recv_task, return_exceptions=True)

        except Exception as exc:
            logger.exception("smallest measure_ttft failed", error=str(exc))
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
        byte_rate = sample_rate * 2  # 16-bit mono
        chunk_size = int(byte_rate * realtime_resolution)
        data = audio_data
        first_chunk = True
        try:
            while data:
                chunk, data = data[:chunk_size], data[chunk_size:]
                if first_chunk:
                    result.audio_start_time = time.monotonic()
                    first_chunk = False
                await ws.send(chunk)
                await asyncio.sleep(realtime_resolution)
            await ws.send(json.dumps({"type": "close_stream"}))
        except Exception as exc:
            logger.exception("smallest send error", error=str(exc))
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_segments: list[str] = []

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()

                transcript = str(msg.get("transcript", "")).strip()
                is_final: bool = bool(msg.get("is_final", False))
                is_last: bool = bool(msg.get("is_last", False))

                if not transcript:
                    if is_last:
                        break
                    continue

                # TTFT — first non-empty transcript (partial or final)
                if result.ttft_seconds is None and result.audio_start_time is not None:
                    result.ttft_seconds = now - result.audio_start_time
                    result.first_token_content = (
                        transcript[:30] + "..." if len(transcript) > 30 else transcript
                    )

                result.partial_transcripts.append(transcript)

                # Accumulate final segments; is_last=true also signals a final segment
                if is_final or is_last:
                    final_segments.append(transcript)
                    if result.audio_start_time is not None:
                        result.audio_to_final_seconds = now - result.audio_start_time

                if is_last:
                    break

        except Exception as exc:
            logger.exception("smallest receive error", error=str(exc))

        if final_segments:
            result.complete_transcript = " ".join(final_segments).strip() or None
        elif result.partial_transcripts:
            result.complete_transcript = max(result.partial_transcripts, key=len).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
