# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseten dedicated-endpoint STT provider (Whisper Large V3 over WebSocket).

The server runs a Silero VAD that consumes audio in fixed 512-sample frames
(32 ms @ 16 kHz). Raw PCM must be sent as binary frames of exactly 512 samples
(1024 bytes); other frame sizes are not processed correctly. The endpoint URL
embeds a private model id, so it is injected from settings rather than hardcoded.
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
from coval_bench.providers.stt._pacing import paced_chunks

logger = structlog.get_logger(__name__)

# Silero VAD frame: 512 samples * 2 bytes = 1024 bytes; other sizes are rejected.
_FRAME_SAMPLES = 512
_FRAME_BYTES = _FRAME_SAMPLES * 2
_BYTE_RATE = 16000 * 2  # 16 kHz mono 16-bit
_MAX_WS_SIZE = 16 * 1024 * 1024
# Cold replicas exceed the 10 s websockets default for the handshake.
_OPEN_TIMEOUT_S = 45


class BasetenSTTProvider(STTProvider):
    """Baseten streaming STT provider (Whisper Large V3)."""

    _VALID_MODELS = frozenset({"whisper-large-v3"})

    def __init__(
        self,
        api_key: SecretStr | None,
        model: str = "whisper-large-v3",
        ws_url: str | None = None,
    ) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Baseten STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None or not api_key.get_secret_value().strip():
            raise ValueError("baseten_api_key is required for the Baseten STT provider")
        if not ws_url:
            raise ValueError("baseten_whisper_url is required for the Baseten STT provider")
        self._api_key = api_key
        self._model = model
        self._ws_url = ws_url

    @property
    def name(self) -> str:
        return "baseten"

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
        if sample_rate != 16000:
            result.error = f"Baseten requires 16 kHz PCM input; got {sample_rate} Hz"
            return result
        if channels != 1 or sample_width != 2:
            result.error = (
                "Baseten requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()
        headers = {"Authorization": f"Api-Key {self._api_key.get_secret_value()}"}

        try:
            async with ws_client.connect(
                self._ws_url,
                additional_headers=headers,
                max_size=_MAX_WS_SIZE,
                open_timeout=_OPEN_TIMEOUT_S,
            ) as ws:
                # First message configures the stream: partials at a 300 ms cadence.
                await ws.send(
                    json.dumps(
                        {
                            "streaming_params": {
                                "enable_partial_transcripts": True,
                                "partial_transcript_interval_s": 0.3,
                            }
                        }
                    )
                )

                send_task = asyncio.create_task(self._send_audio(ws, audio_data, result))
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
                    else:
                        result.error = (
                            "Baseten stream ended before a final transcription was received"
                        )

        except Exception as exc:
            logger.warning(
                "baseten_measure_ttft_failed", provider="baseten", model=self._model, exc_info=exc
            )
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        result: TranscriptionResult,
    ) -> None:
        try:
            async for chunk, start in paced_chunks(audio_data, _FRAME_BYTES, _BYTE_RATE):
                # Stop early if _receive already recorded a protocol/auth error.
                if result.error is not None:
                    break
                result.audio_start_time = start
                frame = chunk
                if len(frame) < _FRAME_BYTES:  # pad the final short frame to 512 samples
                    frame += b"\x00" * (_FRAME_BYTES - len(frame))
                await ws.send(frame)
            # Signal end of audio so the server flushes the final transcript.
            await ws.send(json.dumps({"type": "end_audio"}))
        except Exception as exc:
            logger.warning(
                "baseten_send_error", provider="baseten", model=self._model, exc_info=exc
            )
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        # Per-segment finals accumulate in arrival order (Whisper + Silero VAD).
        final_parts: list[str] = []

        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue  # STT replies are JSON; ignore any binary frames

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()

                msg_type = msg.get("type")
                if msg_type == "end_audio":
                    # Sent twice: "acknowledged" (pre-transcripts) then "finished"
                    # (true end). Stop only on "finished"; the session never closes.
                    if msg.get("body", {}).get("status") == "finished":
                        break
                    continue
                if msg_type == "error":
                    result.error = str(msg.get("message") or msg)
                    logger.warning(
                        "baseten_stt_error", provider="baseten", model=self._model, msg=msg
                    )
                    break
                if msg_type != "transcription":
                    continue

                text = " ".join(str(seg.get("text", "")) for seg in msg.get("segments", [])).strip()
                if not text:
                    continue

                if result.ttft_seconds is None and result.audio_start_time is not None:
                    result.ttft_seconds = now - result.audio_start_time
                    result.first_token_content = text[:30] + "..." if len(text) > 30 else text

                if msg.get("is_final"):
                    final_parts.append(text)
                    if result.audio_start_time is not None:
                        result.audio_to_final_seconds = now - result.audio_start_time
                else:
                    result.partial_transcripts.append(text)

        except Exception as exc:
            logger.warning(
                "baseten_receive_error", provider="baseten", model=self._model, exc_info=exc
            )
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        if final_parts:
            result.complete_transcript = " ".join(final_parts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
