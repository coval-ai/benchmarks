# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI gpt-realtime-whisper STT provider.

Known limitation: this model does not support ``turn_detection: server_vad`` — OpenAI
documents that it requires turn detection to be omitted or set to ``null`` (verified
2026-06). Streaming is therefore single-pass manual-commit: deltas stream in real time
(TTFT stays comparable to native streaming ASR), but the final transcript requires an
explicit ``input_audio_buffer.commit`` after all audio is sent, so AudioToFinal carries
an extra post-audio round-trip and is not directly comparable to auto-finalizing
streaming providers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from collections import defaultdict
from typing import Any, cast

import numpy as np
import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr
from scipy import signal

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._transcript_utils import (
    add_partial_transcript,
    finalize_transcript,
    set_first_token,
)

logger = structlog.get_logger(__name__)

_VALID_MODELS = ("gpt-realtime-whisper",)
_WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
_INPUT_SAMPLE_RATE = 24000
_READY_TIMEOUT_S = 30.0
_RECV_TIMEOUT_S = 10.0


class OpenAISTTProvider(STTProvider):
    """OpenAI Realtime API STT provider (gpt-realtime-whisper)."""

    def __init__(self, api_key: SecretStr, model: str = "gpt-realtime-whisper") -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid OpenAI STT model {model!r}. Valid: {_VALID_MODELS}")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"openai-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_session_update(self) -> dict[str, Any]:
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": _INPUT_SAMPLE_RATE},
                        "transcription": {"model": self._model, "language": "en"},
                        # server_vad is rejected for this model; null = manual-commit only.
                        "turn_detection": None,
                    }
                },
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
        if realtime_resolution <= 0:
            raise ValueError("realtime_resolution must be > 0")

        result = TranscriptionResult(provider=self.name)
        total_start = time.monotonic()

        try:
            prepared_audio = self._prepare_audio(audio_data, channels, sample_width, sample_rate)
            headers = {
                "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            }
            async with ws_client.connect(
                _WS_URL,
                additional_headers=headers,
            ) as ws:
                await self._wait_for_session_ready(ws)
                send_task = asyncio.create_task(
                    self._send_audio(ws, prepared_audio, result, realtime_resolution)
                )
                recv_task = asyncio.create_task(self._receive(ws, result))
                tasks = (send_task, recv_task)
                try:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                    for task in done:
                        exc = task.exception()
                        if exc is not None:
                            raise exc
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as exc:
            logger.exception("openai measure_ttft failed", error=str(exc))
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    def _prepare_audio(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
    ) -> bytes:
        if channels != 1:
            raise ValueError(
                f"OpenAI realtime transcription expects mono PCM; got {channels} channels"
            )
        if sample_width != 2:
            raise ValueError(
                f"OpenAI realtime transcription expects 16-bit PCM; got {sample_width}"
            )
        if sample_rate == _INPUT_SAMPLE_RATE:
            return audio_data
        pcm = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        gcd = math.gcd(sample_rate, _INPUT_SAMPLE_RATE)
        resampled = signal.resample_poly(pcm, up=_INPUT_SAMPLE_RATE // gcd, down=sample_rate // gcd)
        clipped = np.clip(np.round(resampled * 32767.0), -32768, 32767).astype(np.int16)
        return cast(bytes, clipped.tobytes())

    async def _wait_for_session_ready(self, ws: Any) -> None:
        await ws.send(json.dumps(self._build_session_update()))
        try:
            async with asyncio.timeout(_READY_TIMEOUT_S):
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT_S)
                    except StopAsyncIteration as exc:
                        raise RuntimeError(
                            f"Did not receive session.updated within {_READY_TIMEOUT_S}s"
                        ) from exc
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)
                    event_type = str(event.get("type", ""))
                    if event_type == "session.updated":
                        return
                    if event_type == "error":
                        msg = event.get("error", {}).get("message", "Unknown OpenAI error")
                        raise RuntimeError(f"OpenAI error during session setup: {msg}")
        except TimeoutError as exc:
            raise RuntimeError(
                f"Did not receive session.updated within {_READY_TIMEOUT_S}s"
            ) from exc

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        result: TranscriptionResult,
        realtime_resolution: float,
    ) -> None:
        bytes_per_second = _INPUT_SAMPLE_RATE * 2
        chunk_size = max(int(bytes_per_second * realtime_resolution), 2)
        first_chunk = True
        try:
            for start in range(0, len(audio_data), chunk_size):
                chunk = audio_data[start : start + chunk_size]
                if first_chunk and chunk:
                    result.audio_start_time = time.monotonic()
                    first_chunk = False
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("utf-8"),
                        }
                    )
                )
                await asyncio.sleep(realtime_resolution)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception as exc:
            logger.exception("openai send error", error=str(exc))
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        partial_by_item: defaultdict[str, str] = defaultdict(str)
        completed_transcript: str | None = None
        last_final_time: float | None = None
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                event: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                event_type = str(event.get("type", ""))

                if event_type == "conversation.item.input_audio_transcription.delta":
                    delta_raw = str(event.get("delta", ""))
                    if not delta_raw.strip():
                        continue
                    item_id = str(event.get("item_id", "default"))
                    partial_by_item[item_id] += delta_raw
                    current = partial_by_item[item_id].strip()
                    set_first_token(result, current, now=now)
                    add_partial_transcript(result, current)
                    continue

                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = str(event.get("transcript", "")).strip()
                    if not transcript:
                        continue
                    set_first_token(result, transcript, now=now)
                    add_partial_transcript(result, transcript)
                    completed_transcript = transcript
                    if result.audio_start_time is not None:
                        last_final_time = now
                    break

                if event_type == "conversation.item.input_audio_transcription.failed":
                    result.error = str(
                        event.get("error", {}).get("message", "OpenAI transcription failed")
                    )
                    break

                if event_type == "error":
                    result.error = str(event.get("error", {}).get("message", "OpenAI error"))
                    break

        except Exception as exc:
            logger.exception("openai receive error", error=str(exc))
            if result.error is None:
                result.error = str(exc)

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        if completed_transcript is None and result.error is None:
            result.error = "ws_closed_without_completed"

        finalize_transcript(
            result,
            explicit_transcript=completed_transcript,
            partial_fallback="longest",
        )
