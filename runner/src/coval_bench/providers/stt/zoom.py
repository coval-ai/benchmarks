# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Zoom Scribe live STT provider (WebSocket)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import jwt
import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr
from websockets.typing import Subprotocol

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks

logger = structlog.get_logger(__name__)

_WS_URL = "wss://api.zoom.us/v2/aiservices/scribe/live"
_SUBPROTOCOL = Subprotocol("live-asr")
_TOKEN_TTL_S = 7200
_EOT_SILENCE_S = 2.0


class ZoomSTTProvider(STTProvider):
    """Zoom Scribe streaming STT provider."""

    _VALID_MODELS = frozenset({"scribe"})

    def __init__(
        self,
        api_key: SecretStr | None,
        api_secret: SecretStr | None,
        model: str = "scribe",
    ) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Zoom STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None:
            raise ValueError("zoom_api_key is required for the Zoom STT provider")
        if api_secret is None:
            raise ValueError("zoom_api_secret is required for the Zoom STT provider")
        self._api_key = api_key
        self._api_secret = api_secret
        self._model = model

    @property
    def name(self) -> str:
        return "zoom"

    @property
    def model(self) -> str:
        return self._model

    def _auth_token(self) -> str:
        # iat back-dated 30 s for clock skew
        iat = int(time.time()) - 30
        return jwt.encode(
            {"iss": self._api_key.get_secret_value(), "iat": iat, "exp": iat + _TOKEN_TTL_S},
            self._api_secret.get_secret_value(),
            algorithm="HS256",
        )

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
            result.error = f"Zoom Scribe requires 16 kHz PCM input; got {sample_rate} Hz"
            return result
        if channels != 1 or sample_width != 2:
            result.error = (
                "Zoom Scribe requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()

        try:
            async with ws_client.connect(
                _WS_URL,
                subprotocols=[_SUBPROTOCOL],
                additional_headers={"Authorization": f"Bearer {self._auth_token()}"},
            ) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "language": "en-US",
                            "audio": {"format": "pcm16"},
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

        except Exception as exc:
            logger.warning(
                "zoom_measure_ttft_failed", provider="zoom", model=self._model, exc_info=exc
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
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, bytes_per_second):
                result.audio_start_time = start
                await ws.send(chunk)
            # Scribe has no force-finalize: session.close drops the pending turn, so
            # feed trailing silence until the endpointer fires (cap; breaks early).
            final_event.clear()
            silence = bytes(int(bytes_per_second * _EOT_SILENCE_S))
            async for chunk, _ in paced_chunks(silence, chunk_size, bytes_per_second):
                if final_event.is_set():
                    break
                await ws.send(chunk)
            await ws.send(json.dumps({"type": "session.close"}))
        except Exception as exc:
            logger.warning("zoom_send_error", provider="zoom", model=self._model, exc_info=exc)
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
                msg_type = msg.get("type")

                if msg_type == "error":
                    error: dict[str, Any] = msg.get("error") or {}
                    if not error.get("fatal"):
                        logger.warning(
                            "zoom_stt_warning", provider="zoom", model=self._model, msg=msg
                        )
                        continue
                    code = error.get("code")
                    result.error = str(error.get("message") or f"Zoom Scribe error (code {code})")
                    logger.warning("zoom_stt_error", provider="zoom", model=self._model, msg=msg)
                    # Unblocks the send task: its next frame raises ConnectionClosed.
                    await ws.close()
                    break

                if msg_type == "transcription.completed":
                    final_event.set()
                    text = str(msg.get("transcript", "")).strip()
                    if text:
                        if result.ttft_seconds is None and result.audio_start_time is not None:
                            result.ttft_seconds = now - result.audio_start_time
                            result.first_token_content = (
                                text[:30] + "..." if len(text) > 30 else text
                            )
                        final_parts.append(text)
                        if result.audio_start_time is not None:
                            result.audio_to_final_seconds = now - result.audio_start_time

                elif msg_type == "session.closed":
                    break

        except Exception as exc:
            logger.warning("zoom_receive_error", provider="zoom", model=self._model, exc_info=exc)
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        if final_parts:
            result.complete_transcript = " ".join(final_parts)

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
