# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Meta Muse Voice Transcribe real-time STT provider.

  wss://api.meta.ai/v1/asr/realtime
Auth: in the first JSON frame ({"authorization": {"accessToken": "Bearer <key>"}});
the Authorization header is ignored. Server acks with {"sessionId": ...}.
Finalize: {"type": "endStream"}, then read until the server closes with 1000.
"""

from __future__ import annotations

import json
import time
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

_WS_URL = "wss://api.meta.ai/v1/asr/realtime"
_ENCODINGS = {16000: "PCM_16KHZ", 24000: "PCM_24KHZ"}


class MetaSTTProvider(STTProvider):
    """Meta Muse Voice Transcribe streaming STT."""

    def __init__(self, api_key: SecretStr | None, model: str = "muse-voice-transcribe-1.0") -> None:
        if api_key is None or not api_key.get_secret_value():
            raise ValueError("meta_api_key is required in Settings")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "meta"

    @property
    def model(self) -> str:
        return self._model

    def _handshake(self, sample_rate: int) -> dict[str, Any]:
        return {
            "authorization": {"accessToken": f"Bearer {self._api_key.get_secret_value()}"},
            "audioEncoding": _ENCODINGS[sample_rate],
            "model": self._model,
            "mode": "PUSH_TO_TALK",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
            "languageBias": ["English"],
        }

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name)
        if sample_rate not in _ENCODINGS:
            result.error = f"Muse Voice Transcribe needs 16 or 24 kHz PCM; got {sample_rate} Hz"
            return result
        if channels != 1 or sample_width != 2:
            result.error = (
                "Muse Voice Transcribe needs mono 16-bit PCM; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()

        try:
            async with ws_client.connect(_WS_URL) as ws:
                await self._wait_for_session_ready(ws, sample_rate)
                await run_stream(
                    result,
                    self._send_audio(ws, audio_data, sample_rate, result, realtime_resolution),
                    self._receive(ws, result),
                    no_final_error="Muse stream closed without a final transcript",
                )

        except Exception as exc:
            logger.warning(
                "meta_measure_ttft_failed", provider="meta", model=self._model, exc_info=exc
            )
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _wait_for_session_ready(self, ws: Any, sample_rate: int) -> None:
        await ws.send(json.dumps(self._handshake(sample_rate)))
        ack: dict[str, Any] = json.loads(await ws.recv())
        if "sessionId" not in ack:
            raise RuntimeError(f"Muse handshake rejected: {ack.get('message', ack)}")

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        sample_rate: int,
        result: TranscriptionResult,
        realtime_resolution: float,
    ) -> None:
        bytes_per_second = sample_rate * 2
        chunk_size = int(bytes_per_second * realtime_resolution)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, bytes_per_second):
                result.audio_start_time = start
                await ws.send(chunk)
            await ws.send(json.dumps({"type": "endStream"}))
        except Exception as exc:
            logger.warning("meta_send_error", provider="meta", model=self._model, exc_info=exc)
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_text: str | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type = msg.get("type")

                if msg_type == "error":
                    result.error = str(msg.get("message") or "Muse stream error")
                    logger.warning("meta_stt_error", provider="meta", model=self._model, msg=msg)
                    await ws.close()
                    break

                if msg_type != "transcript":
                    continue

                text = str(msg.get("transcript", "")).strip()
                set_first_token(result, text, now=now)
                add_partial_transcript(result, text)

                if msg.get("final"):
                    final_text = text
                    if result.audio_start_time is not None:
                        result.audio_to_final_seconds = now - result.audio_start_time

        except Exception as exc:
            logger.warning("meta_receive_error", provider="meta", model=self._model, exc_info=exc)
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        finalize_transcript(result, explicit_transcript=final_text)
