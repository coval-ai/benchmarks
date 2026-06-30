# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""ElevenLabs real-time STT provider (scribe_v2_realtime).

Wire protocol: WebSocket, wss://api.elevenlabs.io/v1/speech-to-text/realtime
Auth: xi-api-key: <key>
Audio: base64-encoded PCM chunks, JSON-wrapped
Close: {"message_type": "input_audio_chunk", "audio_base_64": "", "commit": true}
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

logger = structlog.get_logger(__name__)

_ERROR_MSG_TYPES = frozenset(
    {
        "error",
        "auth_error",
        "quota_exceeded",
        "commit_throttled",
        "rate_limited",
        "unaccepted_terms",
        "queue_overflow",
        "resource_exhausted",
        "session_time_limit_exceeded",
        "input_error",
        "chunk_size_exceeded",
        "insufficient_audio_activity",
        "transcriber_error",
        "scribe_error",
        "scribe_auth_error",
        "scribe_quota_exceeded_error",
        "scribe_throttled_error",
        "scribe_unaccepted_terms_error",
        "scribe_rate_limited_error",
        "scribe_queue_overflow_error",
        "scribe_resource_exhausted_error",
        "scribe_session_time_limit_exceeded_error",
        "scribe_input_error",
        "scribe_chunk_size_exceeded_error",
        "scribe_insufficient_audio_activity_error",
        "scribe_transcriber_error",
    }
)


class ElevenLabsSTTProvider(STTProvider):
    """ElevenLabs real-time STT provider."""

    _VALID_MODELS = frozenset({"scribe_v2_realtime"})

    def __init__(self, api_key: SecretStr, model: str = "scribe_v2_realtime") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid ElevenLabs model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"elevenlabs-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self) -> str:
        return (
            f"wss://api.elevenlabs.io/v1/speech-to-text/realtime"
            f"?model_id={self._model}&audio_format=pcm_16000"
        )

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
            result.error = f"ElevenLabs requires 16 kHz PCM input; got {sample_rate} Hz"
            return result
        total_start = time.monotonic()

        try:
            url = self._build_websocket_url()
            headers = {"xi-api-key": self._api_key.get_secret_value()}

            async with ws_client.connect(url, additional_headers=headers) as ws:
                # Wait for session_started
                try:
                    raw_session = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    session_data: dict[str, Any] = json.loads(raw_session)
                    if session_data.get("message_type") != "session_started":
                        logger.warning(
                            "elevenlabs_unexpected_first_message",
                            msg=session_data,
                        )
                except TimeoutError:
                    logger.warning("elevenlabs_session_started_timeout")
                except Exception as exc:
                    logger.warning(
                        "elevenlabs_session_started_error",
                        provider="elevenlabs",
                        model=self._model,
                        exc_info=exc,
                    )
                    raise

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
                "elevenlabs_measure_ttft_failed",
                provider="elevenlabs",
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
        # ElevenLabs expects base64-encoded PCM chunks in JSON
        bytes_per_second = sample_rate * 2  # 16-bit mono
        chunk_size = int(bytes_per_second * realtime_resolution)
        start = time.monotonic()
        result.audio_start_time = start
        try:
            for chunk_index, i in enumerate(range(0, len(audio_data), chunk_size)):
                chunk = audio_data[i : i + chunk_size]
                b64 = base64.b64encode(chunk).decode("utf-8")
                await ws.send(
                    json.dumps({"message_type": "input_audio_chunk", "audio_base_64": b64})
                )
                if i + chunk_size < len(audio_data):
                    delay = start + (chunk_index + 1) * realtime_resolution - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)

            # Commit / end-of-input signal
            await ws.send(
                json.dumps(
                    {"message_type": "input_audio_chunk", "audio_base_64": "", "commit": True}
                )
            )
        except Exception as exc:
            logger.warning(
                "elevenlabs_send_error", provider="elevenlabs", model=self._model, exc_info=exc
            )
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        committed_parts: list[str] = []

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("message_type", "")

                if msg_type in _ERROR_MSG_TYPES:
                    error_text = msg.get("message", msg.get("error", "Unknown error"))
                    result.error = f"{msg_type}: {error_text}"
                    logger.warning(
                        "elevenlabs_api_error",
                        provider="elevenlabs",
                        model=self._model,
                        msg_type=msg_type,
                        error=error_text,
                    )
                    break

                if msg_type == "partial_transcript":
                    transcript = msg.get("text", "").strip()
                    if transcript:
                        if result.ttft_seconds is None and result.audio_start_time is not None:
                            result.ttft_seconds = now - result.audio_start_time
                            result.first_token_content = transcript
                        result.partial_transcripts.append(transcript)

                elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
                    # Both types carry "text" in the same position.
                    # committed_transcript_with_timestamps arrives when include_timestamps=true,
                    # but handling it defensively prevents silent data loss if the server ever
                    # sends it unexpectedly (account setting, API default change, etc.).
                    transcript = msg.get("text", "").strip()
                    if transcript:
                        committed_parts.append(transcript)
                        if result.audio_start_time is not None:
                            result.audio_to_final_seconds = now - result.audio_start_time

        except Exception as exc:
            logger.warning(
                "elevenlabs_receive_error",
                provider="elevenlabs",
                model=self._model,
                exc_info=exc,
            )
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        if committed_parts:
            result.complete_transcript = " ".join(committed_parts).strip() or None
        elif result.partial_transcripts:
            result.complete_transcript = result.partial_transcripts[-1].strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
