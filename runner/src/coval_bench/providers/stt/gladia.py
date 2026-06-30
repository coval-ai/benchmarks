# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gladia real-time STT provider (Solaria-1).

The only STT provider here that needs an HTTP step before streaming: a POST to
/v2/live returns a pre-authorized WebSocket URL, so auth rides the POST, not the
socket.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
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

_INIT_URL = "https://api.gladia.io/v2/live"
_ENDPOINTING_S = 0.05  # Gladia's accepted floor
_INIT_TIMEOUT_S = 10.0


class GladiaSTTProvider(STTProvider):
    """Gladia real-time STT provider."""

    _VALID_MODELS = frozenset({"solaria-1"})

    def __init__(self, api_key: SecretStr | None, model: str = "solaria-1") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Gladia STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None:
            raise ValueError("gladia_api_key is required for the Gladia STT provider")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"gladia-{self._model}"

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
        total_start = time.monotonic()

        try:
            ws_url = await self._init_session(channels, sample_width, sample_rate)
            async with ws_client.connect(ws_url) as ws:
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
                try:
                    await recv_task
                finally:
                    send_task.cancel()
                    send_outcome = await asyncio.gather(send_task, return_exceptions=True)
                # CancelledError is a BaseException; isinstance keeps only real send failures.
                if isinstance(send_outcome[0], Exception) and result.error is None:
                    result.error = str(send_outcome[0])

        except Exception as exc:
            logger.warning(
                "gladia_measure_ttft_failed", provider="gladia", model=self._model, exc_info=exc
            )
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _init_session(self, channels: int, sample_width: int, sample_rate: int) -> str:
        config: dict[str, Any] = {
            "encoding": "wav/pcm",
            "bit_depth": sample_width * 8,
            "sample_rate": sample_rate,
            "channels": channels,
            "model": self._model,
            "endpointing": _ENDPOINTING_S,
            "language_config": {"languages": ["en"], "code_switching": False},
            "messages_config": {"receive_partial_transcripts": True},
        }
        headers = {"x-gladia-key": self._api_key.get_secret_value()}
        async with httpx.AsyncClient(timeout=_INIT_TIMEOUT_S) as client:
            resp = await client.post(_INIT_URL, headers=headers, json=config)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gladia session init failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        url = resp.json().get("url")
        if not url:
            raise RuntimeError("Gladia session init returned no WebSocket URL")
        return str(url)

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
        start = time.monotonic()
        result.audio_start_time = start
        sent_bytes = 0
        try:
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await ws.send(chunk)
                sent_bytes += len(chunk)
                delay = start + sent_bytes / byte_rate - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            # Stop streaming; the server flushes pending finals and closes (1000).
            await ws.send(json.dumps({"type": "stop_recording"}))
        except Exception as exc:
            logger.warning("gladia_send_error", provider="gladia", model=self._model, exc_info=exc)
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_parts: list[str] = []

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type == "error":
                    result.error = str(msg.get("message") or msg.get("error") or msg)
                    logger.warning(
                        "gladia_stt_error", provider="gladia", model=self._model, msg=msg
                    )
                    break

                if msg_type != "transcript":
                    continue

                data: dict[str, Any] = msg.get("data") or {}
                utterance: dict[str, Any] = data.get("utterance") or {}
                transcript = str(utterance.get("text", "")).strip()
                if not transcript:
                    continue

                set_first_token(result, transcript, now=now)
                add_partial_transcript(result, transcript)
                if data.get("is_final"):
                    final_parts.append(transcript)
                    if result.audio_start_time is not None:
                        result.audio_to_final_seconds = now - result.audio_start_time

        except Exception as exc:
            logger.warning(
                "gladia_receive_error", provider="gladia", model=self._model, exc_info=exc
            )
            if result.error is None:
                result.error = str(exc)

        finalize_transcript(result, final_segments=final_parts, partial_fallback="longest")
