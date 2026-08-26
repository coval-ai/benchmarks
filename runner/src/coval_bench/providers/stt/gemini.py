# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gemini Live API STT provider (WebSocket) for ``gemini-3.5-transcribe-live``.

Distinct from the ``google`` provider, which speaks Cloud Speech v2 with
service-account auth; the Live API is plain JSON-over-WebSocket with API-key
auth. VERBATIM mode is pinned: SMART cleans disfluencies, which would skew WER
against our verbatim references.
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
from coval_bench.providers.stt._transcript_utils import (
    add_partial_transcript,
    finalize_transcript,
    set_first_token,
)

logger = structlog.get_logger(__name__)

_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
_SETUP_TIMEOUT_S = 30.0

# The server never ends the session itself (verified live 2026-08): it holds the
# socket open after ``audioStreamEnd``, and ``generationComplete`` fires per
# utterance, not per session. The sender waits this long for the post-stream-end
# completion before closing, so the close can't race the last final.
_FINAL_WAIT_S = 5.0


def _decode_event(raw: str | bytes) -> dict[str, Any]:
    """The Live API sends JSON in binary frames; accept text frames too."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    event: dict[str, Any] = json.loads(text)
    return event


class GeminiSTTProvider(STTProvider):
    """Gemini Live API streaming transcription provider."""

    _VALID_MODELS = frozenset({"gemini-3.5-transcribe-live"})

    def __init__(
        self, api_key: SecretStr | None, model: str = "gemini-3.5-transcribe-live"
    ) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Gemini STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None:
            raise ValueError("gemini_api_key is required for the Gemini STT provider")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model

    def _build_setup(self) -> dict[str, Any]:
        return {
            "setup": {
                "model": f"models/{self._model}",
                "generationConfig": {"responseModalities": ["TEXT"]},
                "inputAudioTranscription": {
                    "languageCodes": ["en-US"],
                    "mode": "VERBATIM",
                },
            }
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
        if sample_rate != 16000:
            result.error = f"Gemini Live requires 16 kHz PCM input; got {sample_rate} Hz"
            return result
        if channels != 1 or sample_width != 2:
            result.error = (
                "Gemini Live requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()

        try:
            url = f"{_WS_URL}?key={self._api_key.get_secret_value()}"
            async with ws_client.connect(url) as ws:
                await self._wait_for_setup_complete(ws)

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
                "gemini_measure_ttft_failed", provider="gemini", model=self._model, exc_info=exc
            )
            if result.error is None:
                result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _wait_for_setup_complete(self, ws: Any) -> None:
        await ws.send(json.dumps(self._build_setup()))
        try:
            async with asyncio.timeout(_SETUP_TIMEOUT_S):
                while True:
                    try:
                        raw = await ws.recv()
                    except StopAsyncIteration as exc:
                        raise RuntimeError(
                            f"Did not receive setupComplete within {_SETUP_TIMEOUT_S}s"
                        ) from exc
                    event = _decode_event(raw)
                    if "setupComplete" in event:
                        return
                    if "error" in event:
                        message = event["error"].get("message", "Unknown Gemini error")
                        raise RuntimeError(f"Gemini error during setup: {message}")
        except TimeoutError as exc:
            raise RuntimeError(f"Did not receive setupComplete within {_SETUP_TIMEOUT_S}s") from exc

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
                await ws.send(
                    json.dumps(
                        {
                            "realtimeInput": {
                                "audio": {
                                    "data": base64.b64encode(chunk).decode("ascii"),
                                    "mimeType": f"audio/pcm;rate={sample_rate}",
                                }
                            }
                        }
                    )
                )
            # Completions seen mid-audio belong to earlier utterances; only one
            # arriving after this clear can be the stream-end flush.
            final_event.clear()
            await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(final_event.wait(), timeout=_FINAL_WAIT_S)
            await ws.close()
        except Exception as exc:
            logger.warning("gemini_send_error", provider="gemini", model=self._model, exc_info=exc)
            raise

    async def _receive(
        self, ws: Any, result: TranscriptionResult, final_event: asyncio.Event
    ) -> None:
        final_parts: list[str] = []

        try:
            async for raw in ws:
                msg = _decode_event(raw)
                now = time.monotonic()

                if "error" in msg:
                    result.error = str(msg["error"].get("message", "Gemini error"))
                    logger.warning(
                        "gemini_stt_error", provider="gemini", model=self._model, msg=msg
                    )
                    break

                server_content: dict[str, Any] = msg.get("serverContent") or {}

                interim_event = server_content.get("interimInputTranscription") or {}
                interim = str(interim_event.get("text", ""))
                if interim.strip():
                    set_first_token(result, interim, now=now)
                    add_partial_transcript(result, interim)

                final = str((server_content.get("inputTranscription") or {}).get("text", ""))
                if final.strip():
                    set_first_token(result, final, now=now)
                    final_parts.append(final)
                    if result.audio_start_time is not None:
                        result.audio_to_final_seconds = now - result.audio_start_time

                if server_content.get("generationComplete"):
                    final_event.set()

        except Exception as exc:
            logger.warning(
                "gemini_receive_error", provider="gemini", model=self._model, exc_info=exc
            )
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        finalize_transcript(result, final_segments=final_parts, partial_fallback="longest")
