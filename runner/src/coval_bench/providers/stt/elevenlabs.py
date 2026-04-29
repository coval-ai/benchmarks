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

_VALID_MODELS = ("scribe_v1", "scribe_v2_realtime")

_ERROR_MSG_TYPES = frozenset(
    {
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

    def __init__(self, api_key: SecretStr, model: str = "scribe_v2_realtime") -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid ElevenLabs model {model!r}. Valid: {_VALID_MODELS}")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"elevenlabs-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self) -> str:
        return f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id={self._model}"

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
            url = self._build_websocket_url()
            headers = {"xi-api-key": self._api_key.get_secret_value()}

            async with ws_client.connect(url, additional_headers=headers) as ws:
                # Wait for session_started
                try:
                    raw_session = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    session_data: dict[str, Any] = json.loads(raw_session)
                    if session_data.get("message_type") != "session_started":
                        logger.warning(
                            "unexpected first message from elevenlabs",
                            msg=session_data,
                        )
                except TimeoutError:
                    logger.warning("timeout waiting for elevenlabs session_started")
                except Exception as exc:
                    logger.exception("error reading elevenlabs session_started", error=str(exc))
                    raise

                send_task = asyncio.create_task(
                    self._send_audio(ws, audio_data, sample_rate, result, realtime_resolution)
                )
                recv_task = asyncio.create_task(self._receive(ws, result))
                await asyncio.gather(send_task, recv_task, return_exceptions=True)

        except Exception as exc:
            logger.exception("elevenlabs measure_ttft failed", error=str(exc))
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
        result.audio_start_time = time.monotonic()
        try:
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                b64 = base64.b64encode(chunk).decode("utf-8")
                await ws.send(
                    json.dumps({"message_type": "input_audio_chunk", "audio_base_64": b64})
                )
                await asyncio.sleep(realtime_resolution)

            # Commit / end-of-input signal
            await ws.send(
                json.dumps(
                    {"message_type": "input_audio_chunk", "audio_base_64": "", "commit": True}
                )
            )
        except Exception as exc:
            logger.exception("elevenlabs send error", error=str(exc))
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
                    logger.error(
                        "elevenlabs api error",
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
                    transcript = msg.get("text", "").strip()
                    if transcript:
                        committed_parts.append(transcript)
                        if result.audio_start_time is not None:
                            result.audio_to_final_seconds = now - result.audio_start_time

        except Exception as exc:
            logger.exception("elevenlabs receive error", error=str(exc))

        if committed_parts:
            result.complete_transcript = " ".join(committed_parts).strip() or None
        elif result.partial_transcripts:
            result.complete_transcript = result.partial_transcripts[-1].strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
