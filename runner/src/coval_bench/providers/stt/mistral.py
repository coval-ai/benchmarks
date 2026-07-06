# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Mistral real-time STT provider (Voxtral realtime transcription).

Wire protocol: WebSocket, wss://api.mistral.ai/v1/audio/transcriptions/realtime
Auth: Authorization: Bearer <key>
Close: {"type": "input_audio.flush"} + {"type": "input_audio.end"}, server
replies with transcription.done
"""

from __future__ import annotations

import asyncio
import base64
import json
import time  # monotonic clock — wall-clock can step on NTP sync
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks

logger = structlog.get_logger(__name__)

_WS_BASE = "wss://api.mistral.ai/v1/audio/transcriptions/realtime"
_DEFAULT_MODEL = "voxtral-mini-transcribe-realtime-2602"

# Sample rates accepted by the realtime endpoint for pcm_s16le input.
_SUPPORTED_SAMPLE_RATES = frozenset({8000, 16000, 22050, 44100, 48000})

_SESSION_CREATED_TIMEOUT_S = 10.0


class MistralSTTProvider(STTProvider):
    """Mistral Voxtral realtime transcription provider."""

    _VALID_MODELS = frozenset({_DEFAULT_MODEL})

    def __init__(self, api_key: SecretStr, model: str = _DEFAULT_MODEL) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Mistral model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"mistral-{self._model}"

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
        result = TranscriptionResult(provider=self.name)
        if channels != 1:
            result.error = f"Mistral realtime requires mono audio; got {channels} channels"
            return result
        if sample_width != 2:
            result.error = (
                f"Mistral realtime requires 16-bit PCM (pcm_s16le); got sample_width={sample_width}"
            )
            return result
        if sample_rate not in _SUPPORTED_SAMPLE_RATES:
            result.error = (
                f"Mistral realtime does not support {sample_rate} Hz; "
                f"supported: {sorted(_SUPPORTED_SAMPLE_RATES)}"
            )
            return result
        if realtime_resolution <= 0:
            result.error = "realtime_resolution must be > 0"
            return result

        total_start = time.monotonic()

        try:
            url = f"{_WS_BASE}?model={self._model}"
            headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
            async with ws_client.connect(url, additional_headers=headers) as ws:
                await self._await_session_created(ws)
                # The audio format is rejected once audio has started, so it is
                # set before the first append. target_streaming_delay_ms is left
                # at the server default so latency reflects out-of-box behaviour.
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "audio_format": {
                                    "encoding": "pcm_s16le",
                                    "sample_rate": sample_rate,
                                }
                            },
                        }
                    )
                )
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
                "mistral_measure_ttft_failed",
                provider="mistral",
                model=self._model,
                exc_info=exc,
            )
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _await_session_created(self, ws: Any) -> None:
        deadline = time.monotonic() + _SESSION_CREATED_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for session.created")
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            if isinstance(raw, bytes):
                continue
            msg: dict[str, Any] = json.loads(raw)
            msg_type = msg.get("type", "")
            if msg_type == "session.created":
                return
            if msg_type == "error":
                raise RuntimeError(self._error_message(msg))

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
        chunk_size = int(byte_rate * realtime_resolution)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, byte_rate):
                result.audio_start_time = start
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
            # Flush forces any buffered audio through the model, end signals
            # end-of-stream; the server answers with transcription.done, which
            # the receive loop treats as the final.
            await ws.send(json.dumps({"type": "input_audio.flush"}))
            await ws.send(json.dumps({"type": "input_audio.end"}))
        except Exception as exc:
            logger.warning(
                "mistral_send_error", provider="mistral", model=self._model, exc_info=exc
            )
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        delta_texts: list[str] = []
        final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type == "transcription.text.delta":
                    text = str(msg.get("text", ""))
                    if text.strip():
                        if result.ttft_seconds is None and result.audio_start_time is not None:
                            result.ttft_seconds = now - result.audio_start_time
                            stripped = text.strip()
                            result.first_token_content = (
                                stripped[:30] + "..." if len(stripped) > 30 else stripped
                            )
                        delta_texts.append(text)
                        result.partial_transcripts.append(text)
                elif msg_type == "transcription.done":
                    text = str(msg.get("text", "")).strip()
                    result.complete_transcript = text or None
                    # A done with no text after a delta-less session is a dead
                    # stream, not a final — no audio_to_final from it.
                    if text or delta_texts:
                        final_time = now
                    else:
                        result.error = "Mistral session produced an empty transcript"
                    break
                elif msg_type == "error":
                    result.error = self._error_message(msg)
                    break

        except Exception as exc:
            logger.warning(
                "mistral_receive_error",
                provider="mistral",
                model=self._model,
                exc_info=exc,
            )
            if result.error is None and final_time is None:
                result.error = str(exc)

        if final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = final_time - result.audio_start_time

        if result.complete_transcript is None and delta_texts:
            result.complete_transcript = "".join(delta_texts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())

    def _error_message(self, msg: dict[str, Any]) -> str:
        detail = msg.get("error")
        if isinstance(detail, dict):
            message = detail.get("message")
            if isinstance(message, str) and message:
                return message
            if isinstance(message, dict):
                inner = message.get("detail")
                if isinstance(inner, str) and inner:
                    return inner
        return "Mistral realtime transcription error"
