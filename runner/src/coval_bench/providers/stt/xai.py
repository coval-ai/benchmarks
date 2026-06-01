# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""xAI Grok STT streaming provider."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from urllib.parse import urlencode

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._transcript_utils import (
    add_partial_transcript,
    finalize_transcript,
    set_first_token,
)

logger = structlog.get_logger(__name__)

_VALID_MODELS = ("grok-stt",)
_BASE_WS_URL = "wss://api.x.ai/v1/stt"


class XaiSTTProvider(STTProvider):
    """xAI `/v1/stt` realtime WebSocket provider."""

    def __init__(self, api_key: SecretStr, model: str = "grok-stt") -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid xAI STT model {model!r}. Valid: {_VALID_MODELS}")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"xai-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self, sample_rate: int) -> str:
        params: dict[str, str | int] = {
            "sample_rate": sample_rate,
            "encoding": "pcm",
            "interim_results": "true",
            "endpointing": 200,
            "language": "en",
            "filler_words": "false",
            "diarize": "false",
        }
        return f"{_BASE_WS_URL}?{urlencode(params)}"

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,  # noqa: ARG002
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name)
        if sample_width != 2:
            result.error = (
                f"xAI realtime transcription expects 16-bit PCM input; got {sample_width}"
            )
            return result
        if realtime_resolution <= 0:
            result.error = "realtime_resolution must be > 0"
            return result

        total_start = time.monotonic()

        try:
            headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
            async with ws_client.connect(
                self._build_websocket_url(sample_rate),
                additional_headers=headers,
            ) as ws:
                await self._wait_for_ready(ws)
                send_task = asyncio.create_task(
                    self._send_audio(
                        ws,
                        audio_data,
                        sample_rate,
                        result,
                        realtime_resolution,
                    )
                )
                recv_task = asyncio.create_task(self._receive(ws, result))
                task_results = await asyncio.gather(send_task, recv_task, return_exceptions=True)
                for task_result in task_results:
                    if isinstance(task_result, Exception):
                        raise task_result

        except Exception as exc:
            logger.exception("xai measure_ttft failed", error=str(exc))
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _wait_for_ready(self, ws: Any, *, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"xAI did not signal ready within {timeout:.0f}s")
            raw = await asyncio.wait_for(ws.recv(), timeout=min(5.0, remaining))
            if isinstance(raw, bytes):
                continue

            event: dict[str, Any] = json.loads(raw)
            event_type = str(event.get("type", ""))
            if event_type == "transcript.created":
                return
            if event_type == "error":
                raise RuntimeError(str(event.get("message", "xAI error")))

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        sample_rate: int,
        result: TranscriptionResult,
        realtime_resolution: float,
    ) -> None:
        frame_bytes = 2
        chunk_size = max(frame_bytes, int(sample_rate * frame_bytes * realtime_resolution))
        first_chunk = True

        for start in range(0, len(audio_data), chunk_size):
            chunk = audio_data[start : start + chunk_size]
            if first_chunk and chunk:
                result.audio_start_time = time.monotonic()
                first_chunk = False
            await ws.send(chunk)
            await asyncio.sleep(realtime_resolution)

        await ws.send(json.dumps({"type": "audio.done"}))

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        # xAI is_final=True partials are cumulative restarts, not segments to join.
        last_final_text: str | None = None
        done_transcript: str | None = None
        last_final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                event: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                event_type = str(event.get("type", ""))

                if event_type == "transcript.partial":
                    transcript = str(event.get("text", "")).strip()
                    if not transcript:
                        continue

                    set_first_token(result, transcript, now=now)
                    add_partial_transcript(result, transcript)
                    if event.get("is_final"):
                        last_final_text = transcript
                        if result.audio_start_time is not None:
                            last_final_time = now
                    continue

                if event_type == "transcript.done":
                    done_transcript = str(event.get("text", "")).strip() or None
                    if done_transcript:
                        set_first_token(result, done_transcript, now=now)
                        add_partial_transcript(result, done_transcript)
                    if result.audio_start_time is not None:
                        last_final_time = now
                    break

                if event_type == "error":
                    result.error = str(event.get("message", "xAI error"))
                    break
        except Exception as exc:
            logger.exception("xai receive error", error=str(exc))
            if result.error is None:
                result.error = str(exc)

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        finalize_transcript(
            result,
            # transcript.done text is authoritative when present; otherwise use the last is_final.
            explicit_transcript=done_transcript or last_final_text,
            partial_fallback="longest",
        )
