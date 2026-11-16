# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Murf Finch real-time STT provider.

One WebSocket per clip at ``wss://us-east.api.murf.ai/v1/transcription/stream-input``
(Finch-Global), keyed by an ``api-key`` header. An ``AudioConfig`` text frame
opens the stream, then binary PCM. Turns run in manual commit mode, so the clip's
single turn is closed with ``Finalize`` and the session with ``CloseStream``.
Server messages are ``partial`` (whole turn so far) and ``final`` JSON frames;
an ``Error`` frame always precedes a server close.
"""

from __future__ import annotations

import json
from typing import Any

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_URL = "wss://us-east.api.murf.ai/v1/transcription/stream-input"


class MurfSTTProvider(WebSocketSTTProvider):
    """Murf Finch streaming STT provider."""

    def __init__(self, settings: Settings, model: str = "finch_global_1.0") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("murfai_api_key")

    @property
    def name(self) -> str:
        return f"murf-{self._model}"

    def _ws_url(self, sample_rate: int) -> str:
        return f"{_WS_URL}?model={self._model}&language=en&commit_mode=manual"

    def _headers(self) -> dict[str, str]:
        return {"api-key": self._api_key.get_secret_value()}

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "AudioConfig",
                    "codec": "pcm_s16le",
                    "sample_rate": sample_rate,
                    "channels": 1,
                }
            )
        )

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await self._send_and_wait(ws, json.dumps({"type": "Finalize"}), tx)
        await ws.send(json.dumps({"type": "CloseStream"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type: str = msg.get("type", "")
        text = str(msg.get("transcript") or "").strip()

        if msg_type == "partial":
            if text:
                tx.partial(text, now)
        elif msg_type == "final":
            tx.final(text, now)
        elif msg_type == "Error":
            self._log.warning("murf_stt_error", msg=msg)
            tx.fail(str(msg.get("message") or "Murf STT error"))
            return True
        return False
