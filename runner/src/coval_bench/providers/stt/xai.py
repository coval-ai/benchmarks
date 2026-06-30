# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""xAI Grok STT streaming provider."""

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
from coval_bench.providers.stt._transcript_utils import (
    add_partial_transcript,
    finalize_transcript,
    set_first_token,
)

logger = structlog.get_logger(__name__)

_VALID_MODELS = ("grok-stt",)
_BASE_WS_URL = "wss://api.x.ai/v1/stt"
_FINAL_WAIT_S = 5.0


def _merge_done_with_finals(final_segments: list[str], done_transcript: str) -> list[str]:
    """Reconcile a non-empty transcript.done with the speech_final segments.

    xAI documents transcript.done as the final transcript after audio.done, not
    strictly the still-unclosed tail. It may therefore arrive as the unclosed
    tail only, the whole utterance, or the whole utterance plus a trailing tail.
    When done already contains the joined finalized text as a leading prefix it
    IS the authoritative full transcript, so use it verbatim — appending would
    duplicate (≈100% WER). Otherwise treat done as a trailing tail and append.
    """
    joined = " ".join(" ".join(final_segments).split())
    done = " ".join(done_transcript.split())
    if not joined or done == joined or done.startswith(joined + " "):
        return [done_transcript]
    return [*final_segments, done_transcript]


class XaiSTTProvider(STTProvider):
    """xAI `/v1/stt` realtime WebSocket provider."""

    def __init__(self, api_key: SecretStr, model: str = "grok-stt") -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid xAI STT model {model!r}. Valid: {_VALID_MODELS}")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"xai-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self, sample_rate: int) -> str:
        params: dict[str, str | int] = {
            "sample_rate": sample_rate,
            "encoding": "pcm",
            "interim_results": "true",
            "endpointing": 200,
            "language": "en",
            "filler_words": "true",
            "diarize": "false",
        }
        return f"{_BASE_WS_URL}?{urlencode(params)}"

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,  # noqa: ARG002
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name)
        if sample_width != 2:
            result.error = (
                f"xAI realtime transcription expects 16-bit PCM input; got {sample_width}"
            )
            return result
        if realtime_resolution <= 0:
            result.error = "realtime_resolution must be > 0"
            return result

        total_start = time.monotonic()

        try:
            headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
            async with ws_client.connect(
                self._build_websocket_url(sample_rate),
                additional_headers=headers,
            ) as ws:
                await self._wait_for_ready(ws)
                final_event = asyncio.Event()
                send_task = asyncio.create_task(
                    self._send_audio(
                        ws,
                        audio_data,
                        sample_rate,
                        result,
                        realtime_resolution,
                        final_event,
                    )
                )
                recv_task = asyncio.create_task(self._receive(ws, result, final_event))
                task_results = await asyncio.gather(send_task, recv_task, return_exceptions=True)
                for task_result in task_results:
                    if isinstance(task_result, Exception):
                        raise task_result

        except Exception as exc:
            logger.warning(
                "xai_measure_ttft_failed", provider="xai", model=self._model, exc_info=exc
            )
            result.error = result.error or str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _wait_for_ready(self, ws: Any, *, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"xAI did not signal ready within {timeout:.0f}s")
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(5.0, remaining))
            except TimeoutError:
                continue
            if isinstance(raw, bytes):
                continue

            event: dict[str, Any] = json.loads(raw)
            event_type = str(event.get("type", ""))
            if event_type == "transcript.created":
                return
            if event_type == "error":
                raise RuntimeError(str(event.get("message", "xAI error")))

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        sample_rate: int,
        result: TranscriptionResult,
        realtime_resolution: float,
        final_event: asyncio.Event,
    ) -> None:
        frame_bytes = 2
        chunk_size = max(frame_bytes, int(sample_rate * frame_bytes * realtime_resolution))
        start: float | None = None

        for chunk_index, offset in enumerate(range(0, len(audio_data), chunk_size)):
            chunk = audio_data[offset : offset + chunk_size]
            if start is None and chunk:
                start = time.monotonic()
                result.audio_start_time = start
            await ws.send(chunk)
            if start is not None and offset + chunk_size < len(audio_data):
                delay = start + (chunk_index + 1) * realtime_resolution - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)

        await ws.send(json.dumps({"type": "audio.done"}))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(final_event.wait(), timeout=_FINAL_WAIT_S)
        await ws.close()

    async def _receive(
        self, ws: Any, result: TranscriptionResult, final_event: asyncio.Event
    ) -> None:
        # With endpointing enabled, xAI restarts the transcript at each endpoint:
        # every speech segment is closed exactly once by a speech_final=true
        # partial. The full utterance is the in-order join of speech_final
        # segments. transcript.done is documented only as "final transcript
        # after audio.done" — not guaranteed to be the unclosed tail — so it
        # may be the tail, the whole utterance, or the whole utterance plus a
        # tail; _merge_done_with_finals reconciles it without duplicating.
        # is_final=true/speech_final=false events are interim duplicates of the
        # segment and must NOT be joined.
        final_segments: list[str] = []
        done_transcript: str | None = None
        last_final_time: float | None = None
        done_received = False

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                event: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                event_type = str(event.get("type", ""))

                if event_type == "transcript.partial":
                    transcript = str(event.get("text", "")).strip()
                    if not transcript:
                        continue

                    set_first_token(result, transcript, now=now)
                    add_partial_transcript(result, transcript)
                    if event.get("speech_final"):
                        final_segments.append(transcript)
                        if result.audio_start_time is not None:
                            last_final_time = now
                    continue

                if event_type == "transcript.done":
                    done_received = True
                    done_transcript = str(event.get("text", "")).strip() or None
                    if done_transcript:
                        set_first_token(result, done_transcript, now=now)
                        add_partial_transcript(result, done_transcript)
                        prev_text = " ".join(" ".join(final_segments).split())
                        final_segments = _merge_done_with_finals(final_segments, done_transcript)
                        new_text = " ".join(" ".join(final_segments).split())
                        if new_text != prev_text and result.audio_start_time is not None:
                            last_final_time = now
                    break

                if event_type == "error":
                    result.error = str(event.get("message", "xAI error"))
                    break
        except Exception as exc:
            logger.warning("xai_receive_error", provider="xai", model=self._model, exc_info=exc)
            if result.error is None:
                result.error = str(exc)
        finally:
            final_event.set()

        if not done_received:
            if result.error is None:
                result.error = "xAI stream ended before transcript.done"
            return

        finalize_transcript(
            result,
            final_segments=final_segments,
            partial_fallback="longest",
        )

        # An empty transcript.done that closed no speech_final segment means the
        # session produced nothing (degraded/dead session) — fail it so the row is
        # retryable, instead of stamping a spurious final from the empty terminal.
        if result.complete_transcript is None:
            if result.error is None:
                result.error = "xAI returned an empty transcript"
            return

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time
