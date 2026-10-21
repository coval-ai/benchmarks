# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gladia real-time STT provider (Solaria-1).

The only STT provider here that needs an HTTP step before streaming: a POST to
/v2/live returns a pre-authorized WebSocket URL, so auth rides the POST, not the
socket.
"""

from __future__ import annotations

import json
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_INIT_URL = "https://api.gladia.io/v2/live"
_ENDPOINTING_S = 0.05  # Gladia's accepted floor
_INIT_TIMEOUT_S = 10.0


class GladiaSTTProvider(WebSocketSTTProvider):
    """Gladia real-time STT provider."""

    _PARTIAL_FALLBACK = "longest"

    def __init__(self, settings: Settings, model: str = "solaria-1") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("gladia_api_key")

    @property
    def name(self) -> str:
        return f"gladia-{self._model}"

    async def _connect(self, sample_rate: int) -> AbstractAsyncContextManager[Any]:
        config: dict[str, Any] = {
            "encoding": "wav/pcm",
            "bit_depth": 16,
            "sample_rate": sample_rate,
            "channels": 1,
            "model": self._model,
            "endpointing": _ENDPOINTING_S,
            "language_config": {"languages": ["en"], "code_switching": False},
            "messages_config": {"receive_partial_transcripts": True},
        }
        headers = {"x-gladia-key": self._api_key.get_secret_value()}
        async with httpx.AsyncClient(timeout=_INIT_TIMEOUT_S) as client:
            resp = await client.post(_INIT_URL, headers=headers, json=config)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gladia session init failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        url = resp.json().get("url")
        if not url:
            raise RuntimeError("Gladia session init returned no WebSocket URL")
        return ws_client.connect(str(url))

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        # Stop streaming; the server flushes pending finals and closes (1000).
        await ws.send(json.dumps({"type": "stop_recording"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type: str = msg.get("type", "")
        if msg_type == "error":
            tx.fail(str(msg.get("message") or msg.get("error") or msg))
            self._log.warning("gladia_stt_error", msg=msg)
            return True
        if msg_type != "transcript":
            return False

        data: dict[str, Any] = msg.get("data") or {}
        utterance: dict[str, Any] = data.get("utterance") or {}
        transcript = str(utterance.get("text", "")).strip()
        if not transcript:
            return False

        tx.partial(transcript, now)
        if data.get("is_final"):
            tx.final(transcript, now)
        return False
