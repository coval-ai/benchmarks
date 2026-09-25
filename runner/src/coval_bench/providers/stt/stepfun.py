# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""StepFun realtime STT provider (WebSocket).

StepFun serves StepAudio ASR behind a Realtime-shaped bidirectional socket. The
session runs without ``turn_detection`` so the server performs no VAD: audio
streams in as base64 PCM16 appends and ``input_audio_buffer.commit`` finalizes
the buffer. Deltas carry the cumulative hypothesis (``text`` plus a correctable
``stash`` tail), not increments.
Protocol: https://platform.stepfun.ai/docs/en/api-reference/audio/asr-stream
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from itertools import count
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks
from coval_bench.providers.stt._stream import run_stream
from coval_bench.providers.stt._transcript_utils import (
    add_partial_transcript,
    finalize_transcript,
    set_first_token,
)

logger = structlog.get_logger(__name__)

_WS_URL = "wss://api.stepfun.ai/v1/realtime/asr/stream"
_SAMPLE_RATE = 16000
_READY_TIMEOUT_S = 30.0
_RECV_TIMEOUT_S = 10.0
_DELTA = "conversation.item.input_audio_transcription.delta"
_COMPLETED = "conversation.item.input_audio_transcription.completed"


class _TranscriptionAborted(Exception):
    pass


class StepfunSTTProvider(STTProvider):
    """StepFun streaming STT provider for the StepAudio ASR stream models."""

    def __init__(self, api_key: SecretStr | None, model: str) -> None:
        if api_key is None:
            raise ValueError("stepfun_api_key is required for the StepFun STT provider")
        self._api_key = api_key
        self._model = model
        self._event_ids = count(1)

    @property
    def name(self) -> str:
        return "stepfun"

    @property
    def model(self) -> str:
        return self._model

    def _event(self, event_type: str, **fields: Any) -> str:
        return json.dumps(
            {"event_id": f"event_{next(self._event_ids)}", "type": event_type, **fields}
        )

    def _build_session_update(self) -> str:
        return self._event(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "format": {
                            "type": "pcm",
                            "codec": "pcm_s16le",
                            "rate": _SAMPLE_RATE,
                            "bits": 16,
                            "channel": 1,
                        },
                        "transcription": {"model": self._model, "language": "en"},
                    }
                }
            },
        )

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
            result.error = f"StepFun requires 16 kHz PCM input; got {sample_rate} Hz"
            return result
        if channels != 1 or sample_width != 2:
            result.error = (
                "StepFun requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()

        try:
            headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
                await self._wait_for_session_updated(ws)
                await run_stream(
                    result,
                    self._send_audio(ws, audio_data, result, realtime_resolution),
                    self._receive(ws, result),
                )

        except Exception as exc:
            logger.warning(
                "stepfun_measure_ttft_failed", provider="stepfun", model=self._model, exc_info=exc
            )
            if result.error is None:
                result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _wait_for_session_updated(self, ws: Any) -> None:
        await ws.send(self._build_session_update())
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
                        raise RuntimeError(
                            f"StepFun error during session setup: {_error_message(event)}"
                        )
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
        bytes_per_second = _SAMPLE_RATE * 2
        chunk_size = max(int(bytes_per_second * realtime_resolution), 2)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, bytes_per_second):
                result.audio_start_time = start
                await ws.send(
                    self._event(
                        "input_audio_buffer.append",
                        audio=base64.b64encode(chunk).decode("utf-8"),
                    )
                )
            await ws.send(self._event("input_audio_buffer.commit"))
        except Exception as exc:
            logger.warning(
                "stepfun_send_error", provider="stepfun", model=self._model, exc_info=exc
            )
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        completed_by_item: dict[str, str] = {}
        saw_commit = False
        last_final_time: float | None = None
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                event: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                event_type = str(event.get("type", ""))

                if event_type == _DELTA:
                    hypothesis = f"{event.get('text', '')}{event.get('stash', '')}".strip()
                    if not hypothesis:
                        continue
                    set_first_token(result, hypothesis, now=now)
                    add_partial_transcript(result, hypothesis)
                    continue

                if event_type == _COMPLETED:
                    item_id = str(event.get("item_id", "default"))
                    transcript = str(event.get("transcript", "")).strip()
                    if transcript:
                        set_first_token(result, transcript, now=now)
                        add_partial_transcript(result, transcript)
                        completed_by_item[item_id] = transcript
                    if result.audio_start_time is not None:
                        last_final_time = now
                    if event.get("reason", "commit") == "commit":
                        saw_commit = True
                        break
                    continue

                if event_type == "error":
                    result.error = _error_message(event)
                    raise _TranscriptionAborted

        except _TranscriptionAborted:
            raise
        except Exception as exc:
            logger.warning(
                "stepfun_receive_error", provider="stepfun", model=self._model, exc_info=exc
            )
            if result.error is None:
                result.error = str(exc)

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        if not saw_commit and result.error is None:
            result.error = "ws_closed_without_completed"

        finalize_transcript(
            result,
            final_segments=list(completed_by_item.values()),
            partial_fallback="longest",
        )


def _error_message(event: dict[str, Any]) -> str:
    error = event.get("error") or {}
    message = str(error.get("message") or "StepFun error")
    code = error.get("code")
    return f"{message} (code={code})" if code else message
