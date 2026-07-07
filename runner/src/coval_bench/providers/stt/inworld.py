# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Inworld AI real-time STT provider (WebSocket bidirectional streaming).

INWORLD_API_KEY is the portal-issued base64 credential; it rides the
``Authorization: Basic <key>`` header verbatim (re-encoding it fails auth).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks

logger = structlog.get_logger(__name__)

_WS_URL = "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional"

# Inworld rejects chunks outside 20-1000 ms with "invalid audio chunk duration".
_MIN_CHUNK_MS = 20

# The server never closes after closeStream; the sender closes client-side.
_FINAL_WAIT_S = 5.0


class InworldSTTProvider(STTProvider):
    """Inworld AI streaming STT provider."""

    _VALID_MODELS = frozenset({"inworld-stt-1"})

    def __init__(self, api_key: SecretStr | None, model: str = "inworld-stt-1") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Inworld STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None or not api_key.get_secret_value().strip():
            raise ValueError("inworld_api_key is required for the Inworld STT provider")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "inworld"

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
            result.error = f"Inworld requires 16 kHz PCM input; got {sample_rate} Hz"
            return result
        if channels != 1 or sample_width != 2:
            result.error = (
                "Inworld requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()
        headers = {"Authorization": f"Basic {self._api_key.get_secret_value()}"}

        try:
            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
                # Inworld routes by a "{provider}/{model}" id; we own the inworld/ namespace.
                await ws.send(
                    json.dumps(
                        {
                            "transcribeConfig": {
                                "modelId": f"inworld/{self._model}",
                                "audioEncoding": "LINEAR16",
                                "sampleRateHertz": sample_rate,
                                "numberOfChannels": 1,
                                "language": "en-US",
                            }
                        }
                    )
                )

                final_event = asyncio.Event()
                send_task = asyncio.create_task(
                    self._send_audio(
                        ws, audio_data, sample_rate, result, realtime_resolution, final_event
                    )
                )
                recv_task = asyncio.create_task(self._receive(ws, result, final_event))
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
                            "Inworld stream ended before a final transcription was received"
                        )

        except Exception as exc:
            logger.warning(
                "inworld_measure_ttft_failed", provider="inworld", model=self._model, exc_info=exc
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
        final_event: asyncio.Event,
    ) -> None:
        bytes_per_second = sample_rate * 2  # 16-bit mono
        chunk_size = int(bytes_per_second * realtime_resolution)
        min_tail_bytes = bytes_per_second * _MIN_CHUNK_MS // 1000
        try:
            async for chunk, start in paced_chunks(
                audio_data, chunk_size, bytes_per_second, min_tail_bytes=min_tail_bytes
            ):
                # Stop early if _receive already recorded a protocol/auth error.
                if result.error is not None:
                    break
                result.audio_start_time = start
                await ws.send(
                    json.dumps({"audioChunk": {"content": base64.b64encode(chunk).decode()}})
                )
            final_event.clear()
            await ws.send(json.dumps({"endTurn": {}}))
            await ws.send(json.dumps({"closeStream": {}}))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(final_event.wait(), timeout=_FINAL_WAIT_S)
            await ws.close()
        except Exception as exc:
            logger.warning(
                "inworld_send_error", provider="inworld", model=self._model, exc_info=exc
            )
            raise

    async def _receive(
        self, ws: Any, result: TranscriptionResult, final_event: asyncio.Event
    ) -> None:
        final_parts: list[str] = []

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()

                if msg.get("error"):
                    result.error = str(msg["error"])
                    logger.warning(
                        "inworld_stt_error", provider="inworld", model=self._model, msg=msg
                    )
                    break

                transcription = msg.get("result", {}).get("transcription")
                if not transcription:
                    continue

                text = str(transcription.get("transcript", ""))
                if not text.strip():
                    continue

                if result.ttft_seconds is None and result.audio_start_time is not None:
                    result.ttft_seconds = now - result.audio_start_time
                    snippet = text.strip()
                    result.first_token_content = (
                        snippet[:30] + "..." if len(snippet) > 30 else snippet
                    )

                if transcription.get("isFinal"):
                    final_parts.append(text.strip())
                    if result.audio_start_time is not None:
                        result.audio_to_final_seconds = now - result.audio_start_time
                    final_event.set()
                else:
                    result.partial_transcripts.append(text.strip())

        except Exception as exc:
            logger.warning(
                "inworld_receive_error", provider="inworld", model=self._model, exc_info=exc
            )
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        if final_parts:
            # Inworld emits whole-phrase final segments; join with spaces.
            result.complete_transcript = " ".join(final_parts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
