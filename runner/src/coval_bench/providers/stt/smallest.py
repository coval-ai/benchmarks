# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Smallest AI Pulse real-time STT provider.

Supports models: default (pulse)
Wire protocol: WebSocket, wss://api.smallest.ai/waves/v1/pulse/get_text
Auth: Authorization: Bearer <key>
Query params: language=en, encoding=linear16, sample_rate=<hz>
Audio: raw binary PCM chunks streamed at realtime pace
Close: {"type": "close_stream"}  — server drains buffer, then sends is_last=true
"""

from __future__ import annotations

import json
from typing import Any

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_BASE_URL = "wss://api.smallest.ai/waves/v1/pulse/get_text"


class SmallestSTTProvider(WebSocketSTTProvider):
    """Smallest AI Pulse streaming STT provider."""

    _PARTIAL_FALLBACK = "longest"

    def __init__(self, settings: Settings, model: str = "pulse") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("smallest_api_key")

    @property
    def name(self) -> str:
        return "smallest"

    def _new_result(self) -> TranscriptionResult:
        return TranscriptionResult(provider=self.name, vad_events_count=0)

    def _ws_url(self, sample_rate: int) -> str:
        return f"{_WS_BASE_URL}?language=en&encoding=linear16&sample_rate={sample_rate}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await ws.send(json.dumps({"type": "close_stream"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        transcript = str(msg.get("transcript", "")).strip()
        is_final = bool(msg.get("is_final", False))
        is_last = bool(msg.get("is_last", False))

        if not transcript:
            return is_last

        tx.partial(transcript, now)
        # is_last=true also signals a final segment.
        if is_final or is_last:
            tx.final(transcript, now)
        return is_last
