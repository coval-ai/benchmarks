# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Nari Labs realtime STT provider (WebSocket).

Nari serves Qwen3-ASR behind an OpenAI-Realtime-shaped transcription socket.
The session runs manual-commit (``turn_detection: null``): audio streams in as
base64 PCM16 appends, then ``input_audio_buffer.commit`` finalizes the buffer.
Recordings longer than the session's ``max_duration_seconds`` (36 s) are split
server-side into several items, each finalized with ``commit_reason:
"max_duration"``; the manual commit produces the last item.
Protocol: https://docs.narilabs.com/api-reference/speech-to-text/realtime/realtime-transcription
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
from coval_bench.providers.stt._pacing import paced_chunks
from coval_bench.providers.stt._transcript_utils import (
    add_partial_transcript,
    finalize_transcript,
    set_first_token,
)

logger = structlog.get_logger(__name__)

_WS_URL = "wss://api.narilabs.com/v1/realtime?intent=transcription"
_SAMPLE_RATE = 16000
_READY_TIMEOUT_S = 30.0
_RECV_TIMEOUT_S = 10.0


class _TranscriptionAborted(Exception):
    pass


class NariSTTProvider(STTProvider):
    """Nari Labs streaming STT provider for the hosted Qwen3-ASR models."""

    _VALID_MODELS = frozenset({"qwen3-asr", "qwen3-asr-fast"})

    def __init__(self, api_key: SecretStr | None, model: str = "qwen3-asr-fast") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Nari STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None:
            raise ValueError("nari_api_key is required for the Nari STT provider")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "nari"

    @property
    def model(self) -> str:
        return self._model

    def _build_session_configure(self) -> dict[str, Any]:
        return {
            "type": "session.configure",
            "session": {"model": self._model, "language": "en", "turn_detection": None},
        }

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        if realtime_resolution <= 0:
            raise ValueError("realtime_resolution must be > 0")

        result = TranscriptionResult(provider=self.name)
        if sample_rate != _SAMPLE_RATE:
            result.error = f"Nari requires 16 kHz PCM input; got {sample_rate} Hz"
            return result
        if channels != 1 or sample_width != 2:
            result.error = (
                "Nari requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()

        try:
            headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
                await self._wait_for_session_ready(ws)
                send_task = asyncio.create_task(
                    self._send_audio(ws, audio_data, result, realtime_resolution)
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
            logger.warning(
                "nari_measure_ttft_failed", provider="nari", model=self._model, exc_info=exc
            )
            if result.error is None:
                result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _wait_for_session_ready(self, ws: Any) -> None:
        await ws.send(json.dumps(self._build_session_configure()))
        try:
            async with asyncio.timeout(_READY_TIMEOUT_S):
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT_S)
                    except StopAsyncIteration as exc:
                        raise RuntimeError(
                            f"Did not receive session.configured within {_READY_TIMEOUT_S}s"
                        ) from exc
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)
                    event_type = str(event.get("type", ""))
                    if event_type == "session.configured":
                        return
                    if event_type == "error":
                        raise RuntimeError(
                            f"Nari error during session setup: {_error_message(event)}"
                        )
        except TimeoutError as exc:
            raise RuntimeError(
                f"Did not receive session.configured within {_READY_TIMEOUT_S}s"
            ) from exc

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        result: TranscriptionResult,
        realtime_resolution: float,
    ) -> None:
        bytes_per_second = _SAMPLE_RATE * 2
        chunk_size = max(int(bytes_per_second * realtime_resolution), 2)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, bytes_per_second):
                result.audio_start_time = start
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("utf-8"),
                        }
                    )
                )
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception as exc:
            logger.warning("nari_send_error", provider="nari", model=self._model, exc_info=exc)
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        completed_by_item: dict[str, str] = {}
        saw_manual_commit = False
        last_final_time: float | None = None
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                event: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                event_type = str(event.get("type", ""))

                if event_type == "transcript.partial":
                    transcript = str(event.get("transcript", "")).strip()
                    if not transcript:
                        continue
                    set_first_token(result, transcript, now=now)
                    add_partial_transcript(result, transcript)
                    continue

                if event_type == "transcript.completed":
                    item_id = str(event.get("item_id", "default"))
                    transcript = str(event.get("transcript", "")).strip()
                    if transcript:
                        set_first_token(result, transcript, now=now)
                        add_partial_transcript(result, transcript)
                        completed_by_item[item_id] = transcript
                    if result.audio_start_time is not None:
                        last_final_time = now
                    if event.get("commit_reason") != "max_duration":
                        saw_manual_commit = True
                        break
                    continue

                if event_type == "input_audio_buffer.commit_empty":
                    result.error = "Nari commit had no pending audio"
                    raise _TranscriptionAborted

                if event_type == "error":
                    result.error = _error_message(event)
                    raise _TranscriptionAborted

        except _TranscriptionAborted:
            raise
        except Exception as exc:
            logger.warning("nari_receive_error", provider="nari", model=self._model, exc_info=exc)
            if result.error is None:
                result.error = str(exc)

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        if not saw_manual_commit and result.error is None:
            result.error = "ws_closed_without_completed"

        finalize_transcript(
            result,
            final_segments=list(completed_by_item.values()),
            partial_fallback="longest",
        )


def _error_message(event: dict[str, Any]) -> str:
    error = event.get("error") or {}
    message = str(error.get("message") or "Nari error")
    request_id = error.get("requestId")
    return f"{message} (requestId={request_id})" if request_id else message
