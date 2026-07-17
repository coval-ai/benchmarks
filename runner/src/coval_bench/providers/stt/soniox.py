# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Soniox real-time STT provider (WebSocket)."""

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

_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


class SonioxSTTProvider(STTProvider):
    """Soniox streaming STT provider."""

    _VALID_MODELS = frozenset({"stt-rt-v4", "stt-rt-v5"})

    def __init__(self, api_key: SecretStr | None, model: str = "stt-rt-v5") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Soniox STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None:
            raise ValueError("soniox_api_key is required for the Soniox STT provider")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "soniox"

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
            result.error = f"Soniox requires 16 kHz PCM input; got {sample_rate} Hz"
            return result
        if channels != 1 or sample_width != 2:
            result.error = (
                "Soniox requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()

        try:
            async with ws_client.connect(_WS_URL) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "api_key": self._api_key.get_secret_value(),
                            "model": self._model,
                            "audio_format": "pcm_s16le",
                            "sample_rate": sample_rate,
                            "num_channels": 1,
                            "language_hints": ["en"],
                        }
                    )
                )

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
                "soniox_measure_ttft_failed", provider="soniox", model=self._model, exc_info=exc
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
        bytes_per_second = sample_rate * 2  # 16-bit mono
        chunk_size = int(bytes_per_second * realtime_resolution)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, bytes_per_second):
                result.audio_start_time = start
                await ws.send(chunk)
            # End-of-audio is an empty *text* frame; a zero-length binary frame is
            # ignored, so the server never finalizes (stalls to its idle timeout).
            await ws.send("")
        except Exception as exc:
            logger.warning("soniox_send_error", provider="soniox", model=self._model, exc_info=exc)
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_parts: list[str] = []

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()

                if msg.get("error_code") or msg.get("error_message"):
                    code = msg.get("error_code")
                    result.error = str(
                        msg.get("error_message") or f"Soniox STT error (code {code})"
                    )
                    logger.warning(
                        "soniox_stt_error", provider="soniox", model=self._model, msg=msg
                    )
                    break

                tokens: list[dict[str, Any]] = msg.get("tokens") or []
                for token in tokens:
                    text = str(token.get("text", ""))
                    if not text.strip():
                        continue
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = now - result.audio_start_time
                        snippet = text.strip()
                        result.first_token_content = (
                            snippet[:30] + "..." if len(snippet) > 30 else snippet
                        )
                    if token.get("is_final"):
                        final_parts.append(text)
                        if result.audio_start_time is not None:
                            result.audio_to_final_seconds = now - result.audio_start_time
                    else:
                        result.partial_transcripts.append(text.strip())

                if msg.get("finished"):
                    break

        except Exception as exc:
            logger.warning(
                "soniox_receive_error", provider="soniox", model=self._model, exc_info=exc
            )
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        if final_parts:
            # Soniox tokens carry their own leading spaces — concatenate, don't " ".join.
            result.complete_transcript = "".join(final_parts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
