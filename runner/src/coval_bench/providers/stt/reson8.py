# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reson8 real-time STT provider (WebSocket)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any
from urllib.parse import urlencode

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

_WS_URL = "wss://api.reson8.dev/v1/speech-to-text/realtime"

# Pinned rather than auto-detected: the docs note detection is weaker on short
# utterances, and the STT corpus is English.
_LANGUAGE = "en"

# Cap on waiting for flush_confirmation before closing client-side.
_FLUSH_WAIT_S = 5.0

_FLUSH_ID = "eos"

# Documented error codes. Reson8 publishes the codes but not the frame carrying
# them, so _error_text tests for non-success rather than one literal shape.
_ERROR_CODES = frozenset(
    {"INVALID_REQUEST", "UNAUTHORIZED", "NOT_FOUND", "PAYLOAD_TOO_LARGE", "INTERNAL_ERROR"}
)


def _error_text(msg: dict[str, Any]) -> str | None:
    """Failure description when *msg* is an error frame, else ``None``."""
    frames = [msg]
    for key in ("error", "details"):
        nested = msg.get(key)
        if isinstance(nested, dict):
            frames.append(nested)

    for frame in frames:
        code = frame.get("code")
        if str(frame.get("type", "")).casefold() != "error" and not (
            isinstance(code, str) and code.upper() in _ERROR_CODES
        ):
            continue
        detail = frame.get("message") or frame.get("detail") or code
        return str(detail) if detail else "Reson8 STT error"

    error = msg.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    return None


class Reson8STTProvider(STTProvider):
    """Reson8 streaming STT provider."""

    _VALID_MODELS = frozenset({"realtime"})

    def __init__(self, api_key: SecretStr | None, model: str = "realtime") -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Reson8 STT model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if api_key is None or not api_key.get_secret_value().strip():
            raise ValueError("reson8_api_key is required for the Reson8 STT provider")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "reson8"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self, sample_rate: int) -> str:
        """Connection URL. Auth rides a header, never a query parameter."""
        params = {
            # ``auto`` sniffs container headers; our frames are header-less PCM.
            "encoding": "pcm_s16le",
            "sample_rate": str(sample_rate),
            "channels": "1",
            "language": _LANGUAGE,
            # Without this the server returns finals only, and no ``is_final`` flag.
            "include_interim": "true",
        }
        return f"{_WS_URL}?{urlencode(params)}"

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name)
        if channels != 1 or sample_width != 2:
            result.error = (
                "Reson8 requires mono 16-bit PCM input; "
                f"got channels={channels}, sample_width={sample_width}"
            )
            return result

        total_start = time.monotonic()
        headers = {"Authorization": f"ApiKey {self._api_key.get_secret_value()}"}

        try:
            flushed_event = asyncio.Event()
            async with ws_client.connect(
                self._build_websocket_url(sample_rate), additional_headers=headers
            ) as ws:
                send_task = asyncio.create_task(
                    self._send_audio(
                        ws, audio_data, sample_rate, result, realtime_resolution, flushed_event
                    )
                )
                recv_task = asyncio.create_task(self._receive(ws, result, flushed_event))
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
                        # Stamped here, not in _receive: a close-gate message set
                        # mid-stream would stop the sender and mask a real exception.
                        result.error = "ws_closed_without_final_transcript"

        except Exception as exc:
            logger.warning(
                "reson8_measure_ttft_failed", provider="reson8", model=self._model, exc_info=exc
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
        flushed_event: asyncio.Event,
    ) -> None:
        bytes_per_second = sample_rate * 2  # 16-bit mono
        chunk_size = int(bytes_per_second * realtime_resolution)
        try:
            async for chunk, start in paced_chunks(audio_data, chunk_size, bytes_per_second):
                # The server closes on its error path; stop feeding a dead socket.
                if result.error is not None:
                    return
                result.audio_start_time = start
                await ws.send(chunk)
            await ws.send(json.dumps({"type": "flush_request", "id": _FLUSH_ID}))
            # No end-of-stream message exists: the client closes once the flush is
            # acknowledged, which also releases the receive loop.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(flushed_event.wait(), timeout=_FLUSH_WAIT_S)
            await ws.close()
        except Exception as exc:
            logger.warning("reson8_send_error", provider="reson8", model=self._model, exc_info=exc)
            raise

    async def _receive(
        self, ws: Any, result: TranscriptionResult, flushed_event: asyncio.Event
    ) -> None:
        final_parts: list[str] = []
        recent_frames: list[str] = []
        saw_flush_confirmation = False

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                recent_frames.append(str(raw)[:200])
                del recent_frames[:-3]

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()

                error = _error_text(msg)
                if error is not None:
                    result.error = error
                    logger.warning(
                        "reson8_stt_error", provider="reson8", model=self._model, msg=msg
                    )
                    break

                msg_type = str(msg.get("type", ""))

                if msg_type == "flush_confirmation":
                    saw_flush_confirmation = True
                    break

                if msg_type != "transcript":
                    continue

                transcript = str(msg.get("text", "")).strip()
                if not transcript:
                    continue

                set_first_token(result, transcript, now=now)
                # Absent means the server is returning finals only, which is its
                # behaviour when include_interim is off.
                if msg.get("is_final", True):
                    final_parts.append(transcript)
                    if result.audio_start_time is not None:
                        result.audio_to_final_seconds = now - result.audio_start_time
                else:
                    add_partial_transcript(result, transcript)

        except Exception as exc:
            logger.warning(
                "reson8_receive_error", provider="reson8", model=self._model, exc_info=exc
            )
            if result.error is None and result.audio_to_final_seconds is None:
                result.error = str(exc)

        flushed_event.set()

        if result.error is None and not saw_flush_confirmation:
            # Not a failure on its own — finals may already be in — but the tail
            # went unflushed, so the transcript can be short.
            logger.warning(
                "reson8_flush_unconfirmed",
                provider="reson8",
                model=self._model,
                had_final=bool(final_parts),
                recent_frames=recent_frames,
            )

        finalize_transcript(result, final_segments=final_parts, partial_fallback="longest")
