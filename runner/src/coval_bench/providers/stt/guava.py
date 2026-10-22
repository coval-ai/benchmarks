# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Guava STT provider (daytona-stt).

Wire protocol: WebSocket, <guava_base_url>/audio/transcriptions (http -> ws, https -> wss).
Auth: Authorization: Bearer <key>.
Start: {"type": "start", "partial_transcripts": true}, with optional "domain".
Audio: raw int16-LE PCM, mono, 16 kHz binary frames.
Close: {"type": "eos"}
Server messages (JSON, keyed by ``type``):
  asr_partial -> ``transcript`` + ``ts_ms``
  asr_final   -> ``transcript`` + ``ts_ms``
  error       -> ``message``
"""

from __future__ import annotations

import json
from contextlib import AbstractAsyncContextManager
from typing import Any

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_PATH = "/audio/transcriptions"
_MAX_WS_SIZE = 16 * 1024 * 1024


def ws_url_from_base(base_url: str) -> str:
    """Derive the ASR WebSocket URL from the shared guava base URL."""
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    return base + _WS_PATH


class GuavaSTTProvider(WebSocketSTTProvider):
    """Guava STT provider"""

    _SAMPLE_RATE = 16000
    _NO_FINAL_ERROR = "Guava stream ended before a final transcription was received"

    def __init__(self, settings: Settings, model: str = "daytona-stt") -> None:
        super().__init__(settings, model)
        self._url = ws_url_from_base(self._require_setting("guava_base_url"))
        self._api_key = self._require_secret("guava_api_key")
        domain = settings.guava_stt_domain
        self._domain = domain.strip() if domain else None

    @property
    def name(self) -> str:
        return "guava"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    async def _connect(self, sample_rate: int) -> AbstractAsyncContextManager[Any]:
        return ws_client.connect(
            self._url, additional_headers=self._headers(), max_size=_MAX_WS_SIZE
        )

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        start_message: dict[str, Any] = {"type": "start", "partial_transcripts": True}
        if self._domain:
            start_message["domain"] = self._domain
        await ws.send(json.dumps(start_message))

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        # The server finalizes and closes after this.
        await ws.send(json.dumps({"type": "eos"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type = msg.get("type")
        if msg_type == "error":
            tx.fail(str(msg.get("message") or msg))
            self._log.warning("guava_stt_error", msg=msg)
            return True
        if msg_type not in ("asr_partial", "asr_final"):
            return False

        text = str(msg.get("transcript", "")).strip()
        if msg_type == "asr_final":
            tx.final(text, now)
        else:
            tx.partial(text, now)
        return False
