# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Soniox real-time STT provider.

Supports models: stt-rt-v4
Wire protocol: WebSocket, wss://stt-rt.soniox.com/transcribe-websocket
Auth: in-band — the api_key rides the opening JSON config frame (no header).
Config: {"api_key":..., "model":"stt-rt-v4", "audio_format":"pcm_s16le",
         "sample_rate":16000, "num_channels":1, "language_hints":["en"]}
Audio: raw PCM_16 sent as binary frames.
End: an empty text frame (`""`) signals end of audio; the server flushes final
     tokens and sends {"finished": true} before closing. A zero-length binary
     frame is ignored, so the stream must be ended with the empty string.

Protocol notes:
- Responses carry a "tokens" array; each token has "text" and "is_final".
  Final tokens (is_final=True) are committed and never change; non-final tokens
  are interim and may be superseded by later messages.
- Token "text" already carries its own spacing, so the transcript is the direct
  concatenation of final-token text (not a " ".join()).
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

logger = structlog.get_logger(__name__)

_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


class SonioxSTTProvider(STTProvider):
    """Soniox streaming STT provider."""

    _VALID_MODELS = frozenset({"stt-rt-v4"})

    def __init__(self, api_key: SecretStr | None, model: str = "stt-rt-v4") -> None:
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

        total_start = time.monotonic()

        try:
            async with ws_client.connect(_WS_URL) as ws:
                # Soniox authenticates in-band: the api_key rides the opening config
                # frame rather than an Authorization header.
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
                await asyncio.gather(send_task, recv_task, return_exceptions=True)

        except Exception as exc:
            logger.exception("soniox measure_ttft failed", error=str(exc))
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
        result.audio_start_time = time.monotonic()
        try:
            for i in range(0, len(audio_data), chunk_size):
                await ws.send(audio_data[i : i + chunk_size])
                await asyncio.sleep(realtime_resolution)
            # An empty *text* frame signals end of audio; the server then flushes
            # final tokens and sends {"finished": true}. A zero-length binary frame
            # is ignored, leaving the stream open until the server's idle timeout.
            await ws.send("")
        except Exception as exc:
            logger.exception("soniox send error", error=str(exc))
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
                    logger.error("soniox stt error", msg=msg)
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
            logger.exception("soniox receive error", error=str(exc))

        if final_parts:
            result.complete_transcript = "".join(final_parts).strip() or None

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
