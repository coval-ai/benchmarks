# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""xAI Grok STT streaming provider."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlencode

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_BASE_WS_URL = "wss://api.x.ai/v1/stt"
_READY_TIMEOUT_S = 10.0


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


@dataclass
class _XaiTranscription(Transcription):
    done_received: bool = False


class XaiSTTProvider(WebSocketSTTProvider):
    """xAI `/v1/stt` realtime WebSocket provider."""

    _PARTIAL_FALLBACK = "longest"

    def __init__(self, settings: Settings, model: str = "grok-stt") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("xai_api_key")

    @property
    def name(self) -> str:
        return f"xai-{self._model}"

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _XaiTranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
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

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        try:
            async with asyncio.timeout(_READY_TIMEOUT_S):
                while True:
                    try:
                        raw = await ws.recv()
                    except StopAsyncIteration:
                        raise RuntimeError("xAI stream closed before transcript.created") from None
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)
                    event_type = str(event.get("type", ""))
                    if event_type == "transcript.created":
                        return
                    if event_type == "error":
                        raise RuntimeError(str(event.get("message", "xAI error")))
        except TimeoutError as exc:
            raise RuntimeError(f"xAI did not signal ready within {_READY_TIMEOUT_S:.0f}s") from exc

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        # The latch here is transcript.done, so the close can't race it.
        await self._send_and_wait(ws, json.dumps({"type": "audio.done"}), tx)
        await ws.close()

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        # With endpointing enabled, xAI restarts the transcript at each endpoint:
        # every speech segment is closed exactly once by a speech_final=true
        # partial, and the utterance is their in-order join. is_final=true /
        # speech_final=false events are interim duplicates and are never joined.
        tx = cast(_XaiTranscription, tx)
        event_type = str(msg.get("type", ""))

        if event_type == "transcript.partial":
            transcript = str(msg.get("text", "")).strip()
            if not transcript:
                return False
            tx.partial(transcript, now)
            if msg.get("speech_final"):
                tx.final(transcript, now)
            return False

        if event_type == "transcript.done":
            tx.done_received = True
            done = str(msg.get("text", "")).strip()
            if done:
                tx.partial(done, now)
                before = " ".join(" ".join(tx.finals).split())
                tx.finals[:] = _merge_done_with_finals(tx.finals, done)
                if " ".join(" ".join(tx.finals).split()) != before:
                    tx.final("", now)
            tx.final_event.set()
            return True

        if event_type == "error":
            tx.fail(str(msg.get("message", "xAI error")))
            self._log.warning("xai_stt_error", msg=msg)
            return True

        return False

    async def _receive(self, ws: Any, tx: Transcription) -> None:
        # Post-hook only: a server close without transcript.done is a failure.
        await super()._receive(ws, tx)
        if not cast(_XaiTranscription, tx).done_received:
            tx.fail("xAI stream ended before transcript.done")

    def _finish(self, tx: Transcription) -> None:
        tx = cast(_XaiTranscription, tx)
        if not tx.done_received:
            tx.result.audio_to_final_seconds = None
            return
        super()._finish(tx)
        # An empty transcript.done that closed no speech_final segment means the
        # session produced nothing (degraded/dead session); fail it so the row is
        # retryable, instead of stamping a spurious final from the empty terminal.
        if tx.result.complete_transcript is None:
            tx.result.audio_to_final_seconds = None
            tx.fail("xAI returned an empty transcript")
