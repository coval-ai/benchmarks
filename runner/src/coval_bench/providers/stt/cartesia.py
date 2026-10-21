# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cartesia Ink real-time STT provider (ink-2).

The auto-finalize endpoint (/stt/turns/websocket, turn.* events) is intentionally
not used: its model-driven turn segmentation diverges from how every other STT
provider here is benchmarked (single stream, progressive finals, finalize-on-close).
"""

from __future__ import annotations

from typing import Any

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_BASE_URL = "wss://api.cartesia.ai/stt/websocket"
_CARTESIA_VERSION = "2025-11-04"


class CartesiaSTTProvider(WebSocketSTTProvider):
    """Cartesia Ink streaming STT provider."""

    # Deltas carry their own leading spaces; joining with " " would double them.
    _FINAL_JOIN = ""

    def __init__(self, settings: Settings, model: str = "ink-2") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("cartesia_api_key")

    @property
    def name(self) -> str:
        return f"cartesia-{self._model}"

    def _new_result(self) -> TranscriptionResult:
        return TranscriptionResult(provider=self.name, vad_events_count=0)

    def _ws_url(self, sample_rate: int) -> str:
        return (
            f"{_WS_BASE_URL}"
            f"?model={self._model}"
            f"&encoding=pcm_s16le"
            f"&sample_rate={sample_rate}"
            f"&language=en"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "cartesia-version": _CARTESIA_VERSION,
        }

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        # Two-phase end: finalize flushes buffered audio; close ends the session.
        await ws.send("finalize")
        await ws.send("close")

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type: str = msg.get("type", "")
        if msg_type == "error":
            tx.fail(str(msg.get("message") or msg))
            return True
        # flush_done acks a finalize; done is the terminal marker. Break on
        # done explicitly rather than relying on the socket closing.
        if msg_type == "done":
            return True
        if msg_type != "transcript":
            return False

        text = str(msg.get("text", ""))
        if not text:
            return False
        # A standalone " " delta is a word separator: no TTFT or partial, but
        # it must reach the transcript or neighbouring words are glued.
        tx.partial(text, now)
        if msg.get("is_final"):
            tx.final(text, now)
        return False
