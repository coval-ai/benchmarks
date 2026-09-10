# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Guava STT provider (daytona-stt).

Wire protocol: WebSocket, <guava_base_url>/audio/transcriptions (http -> ws, https -> wss).
Auth: Authorization: Bearer <key>.
Start: {"type": "start", "domain": <domain>, "partial_transcripts": true}
Audio: raw int16-LE PCM, mono, 16 kHz binary frames.
Close: {"type": "eos"}
Server messages (JSON, keyed by ``type``):
  asr_partial -> ``transcript`` + ``ts_ms``
  asr_final   -> ``transcript`` + ``ts_ms``
  error       -> ``message``
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

_WS_PATH = "/audio/transcriptions"
_SAMPLE_RATE = 16000

_MAX_WS_SIZE = 16 * 1024 * 1024
_NO_FINAL_ERROR = "Guava stream ended before a final transcription was received"


def ws_url_from_base(base_url: str) -> str:
    """Derive the ASR WebSocket URL from the shared guava base URL."""
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    return base + _WS_PATH


class GuavaSTTProvider(STTProvider):
    """Guava STT provider"""

    _VALID_MODELS = frozenset({"daytona-stt"})

    def __init__(
        self,
        api_key: SecretStr | None,
        model: str = "daytona-stt",
        base_url: str | None = None,
        domain: str | None = None,
    ) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Guava STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if not base_url:
            raise ValueError("guava_base_url is required for the Guava STT provider")
        if api_key is None or not api_key.get_secret_value().strip():
            raise ValueError("guava_api_key is required for the Guava STT provider")
        if not domain or not domain.strip():
            raise ValueError("guava_stt_domain is required for the Guava STT provider")
        self._api_key = api_key
        self._model = model
        self._ws_url = ws_url_from_base(base_url)
        self._domain = domain.strip()

    @property
    def name(self) -> str:
        return "guava"

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
        if sample_rate != _SAMPLE_RATE:
            result.error = f"Guava requires 16 kHz PCM input; got {sample_rate} Hz"
            return result
        if channels != 1 or sample_width != 2:
            result.error = (
                "Guava requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

        try:
            async with ws_client.connect(
                self._ws_url,
                additional_headers=headers or None,
                max_size=_MAX_WS_SIZE,
            ) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "start",
                            "domain": self._domain,
                            "partial_transcripts": True,
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
                    else:
                        result.error = _NO_FINAL_ERROR

        except Exception as exc:
            logger.warning(
                "guava_measure_ttft_failed", provider="guava", model=self._model, exc_info=exc
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
        byte_rate = sample_rate * 2  # 16-bit mono
        chunk_size = int(byte_rate * realtime_resolution)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, byte_rate):
                # Stop early if _receive already recorded a protocol/auth error.
                if result.error is not None:
                    break
                result.audio_start_time = start
                await ws.send(chunk)
            # End of audio: the server finalizes and closes after this.
            await ws.send(json.dumps({"type": "eos"}))
        except Exception as exc:
            logger.warning("guava_send_error", provider="guava", model=self._model, exc_info=exc)
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        # Finals accumulate in arrival order.
        final_parts: list[str] = []

        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue  # transcripts are JSON; ignore any binary frames

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()

                msg_type = msg.get("type")
                if msg_type == "error":
                    result.error = str(msg.get("message") or msg)
                    logger.warning("guava_stt_error", provider="guava", model=self._model, msg=msg)
                    break
                if msg_type not in ("asr_partial", "asr_final"):
                    continue

                text = str(msg.get("transcript", "")).strip()
                if text and result.ttft_seconds is None and result.audio_start_time is not None:
                    result.ttft_seconds = now - result.audio_start_time
                    result.first_token_content = text[:30] + "..." if len(text) > 30 else text

                if msg_type == "asr_final":
                    if text:
                        final_parts.append(text)
                    if result.audio_start_time is not None:
                        result.audio_to_final_seconds = now - result.audio_start_time
                elif text:
                    result.partial_transcripts.append(text)

        except Exception as exc:
            logger.warning("guava_receive_error", provider="guava", model=self._model, exc_info=exc)
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        if final_parts:
            result.complete_transcript = " ".join(final_parts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
