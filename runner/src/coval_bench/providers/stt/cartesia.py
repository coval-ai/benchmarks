# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cartesia Ink real-time STT provider (ink-2).

The auto-finalize endpoint (/stt/turns/websocket, turn.* events) is intentionally
not used: its model-driven turn segmentation diverges from how every other STT
provider here is benchmarked (single stream, progressive finals, finalize-on-close).
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

_VALID_MODELS = ("ink-2",)
_WS_BASE_URL = "wss://api.cartesia.ai/stt/websocket"
_CARTESIA_VERSION = "2025-11-04"


class CartesiaSTTProvider(STTProvider):
    """Cartesia Ink streaming STT provider."""

    def __init__(self, api_key: SecretStr, model: str = "ink-2") -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Cartesia STT model {model!r}. Valid: {_VALID_MODELS}")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"cartesia-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self, sample_rate: int) -> str:
        return (
            f"{_WS_BASE_URL}"
            f"?model={self._model}"
            f"&encoding=pcm_s16le"
            f"&sample_rate={sample_rate}"
            f"&language=en"
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
        if channels != 1:
            raise ValueError(
                f"Cartesia Ink only supports mono audio (channels=1), got channels={channels}"
            )
        if sample_width != 2:
            raise ValueError(
                f"Cartesia Ink expects 16-bit PCM (sample_width=2), got sample_width={sample_width}"
            )
        result = TranscriptionResult(provider=self.name, vad_events_count=0)
        total_start = time.monotonic()

        try:
            url = self._build_websocket_url(sample_rate)
            headers = {
                "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                "cartesia-version": _CARTESIA_VERSION,
            }

            async with ws_client.connect(url, additional_headers=headers) as ws:
                send_task = asyncio.create_task(
                    self._send_audio(ws, audio_data, sample_rate, result, realtime_resolution)
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
                "cartesia_stt_measure_ttft_failed",
                provider="cartesia",
                model=self._model,
                exc_info=exc,
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
    ) -> None:
        byte_rate = sample_rate * 2  # 16-bit mono
        chunk_size = int(byte_rate * realtime_resolution)
        data = audio_data
        start: float | None = None
        chunk_index = 0
        try:
            while data:
                chunk, data = data[:chunk_size], data[chunk_size:]
                if start is None:
                    start = time.monotonic()
                    result.audio_start_time = start
                await ws.send(chunk)
                if data:
                    delay = start + (chunk_index + 1) * realtime_resolution - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                chunk_index += 1
            # Two-phase end: finalize flushes buffered audio; close ends the session.
            await ws.send("finalize")
            await ws.send("close")
        except Exception as exc:
            logger.warning(
                "cartesia_stt_send_error", provider="cartesia", model=self._model, exc_info=exc
            )
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_segments: list[str] = []

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type == "error":
                    result.error = str(msg.get("message") or msg)
                    break
                # flush_done acks a finalize; done is the terminal marker. Break on
                # done explicitly rather than relying on the socket closing.
                if msg_type == "done":
                    break
                if msg_type != "transcript":
                    continue

                # Keep the raw text: Cartesia deltas carry their own leading spaces,
                # which are the word separators. Strip only for the empty check + display.
                text = str(msg.get("text", ""))
                token = text.strip()
                if not token:
                    continue

                # TTFT — first non-empty transcript (partial or final).
                if result.ttft_seconds is None and result.audio_start_time is not None:
                    result.ttft_seconds = now - result.audio_start_time
                    result.first_token_content = token[:30] + "..." if len(token) > 30 else token

                result.partial_transcripts.append(token)

                if msg.get("is_final"):
                    final_segments.append(text)
                    if result.audio_start_time is not None:
                        result.audio_to_final_seconds = now - result.audio_start_time

        except Exception as exc:
            logger.warning(
                "cartesia_stt_receive_error",
                provider="cartesia",
                model=self._model,
                exc_info=exc,
            )
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        # Per Cartesia docs, join is_final deltas verbatim — they carry their own
        # spacing; do NOT inject separators between them. Outer strip only, for display.
        # If no is_final arrived, leave complete_transcript None so the orchestrator
        # scores WER as failed rather than from a partial fragment.
        if final_segments:
            result.complete_transcript = "".join(final_segments).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
