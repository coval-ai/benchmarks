# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Template for WebSocket streaming STT providers.

A provider supplies the connection (URL and headers, or a whole ``_connect``),
an optional opening frame, the end-of-audio message, and a parser for one
server message. The base owns pacing, the send/receive tasks, error logging and
the final transcript. Hooks are listed in the order they run.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, ClassVar

import websockets.asyncio.client as ws_client

from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks
from coval_bench.providers.stt._stream import run_stream
from coval_bench.providers.stt._transcript_utils import (
    PartialFallback,
    add_partial_transcript,
    finalize_transcript,
    set_first_token,
)


class StreamAborted(Exception):
    """The server reported an error; raised from the receiver to cancel the sender."""


@dataclass
class Transcription:
    """Per-call receive state shared between the send and receive tasks."""

    result: TranscriptionResult
    finals: list[str] = field(default_factory=list)
    # Set on every final; senders wait on it after the end-of-audio frame.
    final_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Pacing of the audio just sent, for providers that drain trailing silence.
    byte_rate: int = 0
    chunk_size: int = 0

    def partial(self, text: str, now: float) -> None:
        """Record an interim transcript; the first non-empty one sets TTFT."""
        set_first_token(self.result, text, now=now)
        add_partial_transcript(self.result, text)

    def final(self, text: str, now: float) -> None:
        """Record a committed segment and the audio-to-final time."""
        set_first_token(self.result, text, now=now)
        if text:
            self.finals.append(text)
        if self.result.audio_start_time is not None:
            self.result.audio_to_final_seconds = now - self.result.audio_start_time
        self.final_event.set()

    def fail(self, message: str) -> None:
        """Stamp a provider-reported error; the first one wins."""
        if self.result.error is None:
            self.result.error = message


class WebSocketSTTProvider(STTProvider):
    """Streaming STT over one WebSocket: paced binary audio in, JSON messages out."""

    _FINAL_JOIN: ClassVar[str] = " "
    _PARTIAL_FALLBACK: ClassVar[PartialFallback] = "none"
    # Stamped when the stream ends cleanly without any final.
    _NO_FINAL_ERROR: ClassVar[str | None] = None
    # A final chunk smaller than this is rebalanced with its predecessor.
    _MIN_TAIL_BYTES: ClassVar[int] = 0
    # Servers that speak JSON in binary frames set this False.
    _SKIP_BINARY_FRAMES: ClassVar[bool] = True
    # How long _send_and_wait waits for the server's reply to the end-of-audio frame.
    _FINAL_WAIT_S: ClassVar[float] = 5.0

    # -- connection -----------------------------------------------------------

    def _ws_url(self, sample_rate: int) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {}

    async def _connect(self, sample_rate: int) -> AbstractAsyncContextManager[Any]:
        return ws_client.connect(
            self._ws_url(sample_rate), additional_headers=self._headers() or None
        )

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        """Opening handshake: send a config frame, await a ready message, or nothing."""

    # -- send -----------------------------------------------------------------

    def _prepare_audio(self, audio_data: bytes, sample_rate: int) -> tuple[bytes, int]:
        """Return the bytes to stream and their real-time byte rate."""
        return audio_data, sample_rate * 2

    def _chunk_size(self, byte_rate: int, realtime_resolution: float) -> int:
        return max(2, int(byte_rate * realtime_resolution))

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        await ws.send(chunk)

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        """Tell the server the audio is over (and wait for its final, if it has one)."""
        raise NotImplementedError

    async def _send_and_wait(
        self, ws: Any, frame: str | bytes, tx: Transcription, *, event: asyncio.Event | None = None
    ) -> bool:
        """Send *frame*, then wait up to ``_FINAL_WAIT_S`` for *event* (default: a final).

        The shared final latch is cleared first, since a final that landed mid-stream
        would make the wait a no-op; a dedicated ack event is only ever set by its ack.
        Returns False on timeout so the caller can close anyway.
        """
        latch = tx.final_event if event is None else event
        if event is None:
            latch.clear()
        await ws.send(frame)
        try:
            await asyncio.wait_for(latch.wait(), timeout=self._FINAL_WAIT_S)
        except TimeoutError:
            return False
        return True

    async def _send(
        self,
        ws: Any,
        audio_data: bytes,
        sample_rate: int,
        realtime_resolution: float,
        tx: Transcription,
    ) -> None:
        audio, byte_rate = self._prepare_audio(audio_data, sample_rate)
        tx.byte_rate = byte_rate
        tx.chunk_size = self._chunk_size(byte_rate, realtime_resolution)
        try:
            async for chunk, start in paced_chunks(
                audio, tx.chunk_size, byte_rate, min_tail_bytes=self._MIN_TAIL_BYTES
            ):
                tx.result.audio_start_time = start
                await self._send_chunk(ws, chunk)
            await self._end_audio(ws, tx)
        except Exception as exc:
            self._log.warning("stt_send_error", exc_info=exc)
            raise

    # -- receive --------------------------------------------------------------

    def _decode(self, raw: str | bytes) -> Any:
        return json.loads(raw)

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        """Handle one decoded server message. Return True to stop receiving."""
        raise NotImplementedError

    async def _receive(self, ws: Any, tx: Transcription) -> None:
        try:
            async for raw in ws:
                if self._SKIP_BINARY_FRAMES and isinstance(raw, bytes):
                    continue
                if self._on_message(self._decode(raw), time.monotonic(), tx):
                    break
        except Exception as exc:
            self._log.warning("stt_receive_error", exc_info=exc)
            raise
        if tx.result.error is not None:
            raise StreamAborted(tx.result.error)

    def _finish(self, tx: Transcription) -> None:
        finalize_transcript(
            tx.result,
            final_segments=tx.finals,
            partial_fallback=self._PARTIAL_FALLBACK,
            join=self._FINAL_JOIN,
        )

    # -- template -------------------------------------------------------------

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return Transcription(result)

    async def _transcribe(
        self,
        audio_data: bytes,
        sample_rate: int,
        realtime_resolution: float,
        result: TranscriptionResult,
    ) -> None:
        tx = self._new_transcription(result)
        try:
            async with await self._connect(sample_rate) as ws:
                await self._open(ws, tx, sample_rate)
                await run_stream(
                    result,
                    self._send(ws, audio_data, sample_rate, realtime_resolution, tx),
                    self._receive(ws, tx),
                    no_final_error=self._NO_FINAL_ERROR,
                )
        finally:
            # In finally so a cancellation mid-stream still commits received finals.
            self._finish(tx)
