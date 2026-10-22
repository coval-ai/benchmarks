# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Zoom Scribe live STT provider (WebSocket)."""

from __future__ import annotations

import json
import time
from contextlib import AbstractAsyncContextManager
from typing import Any

import jwt
import websockets.asyncio.client as ws_client
from websockets.typing import Subprotocol

from coval_bench.config import Settings
from coval_bench.providers.stt._pacing import paced_chunks
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_URL = "wss://api.zoom.us/v2/aiservices/scribe/live"
_SUBPROTOCOL = Subprotocol("live-asr")
_TOKEN_TTL_S = 7200
_EOT_SILENCE_S = 2.0


class ZoomSTTProvider(WebSocketSTTProvider):
    """Zoom Scribe streaming STT provider."""

    _SAMPLE_RATE = 16000

    def __init__(self, settings: Settings, model: str = "scribe") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("zoom_api_key")
        self._api_secret = self._require_secret("zoom_api_secret")

    @property
    def name(self) -> str:
        return "zoom"

    def _auth_token(self) -> str:
        # iat back-dated 30 s for clock skew
        iat = int(time.time()) - 30
        return jwt.encode(
            {"iss": self._api_key.get_secret_value(), "iat": iat, "exp": iat + _TOKEN_TTL_S},
            self._api_secret.get_secret_value(),
            algorithm="HS256",
        )

    async def _connect(self, sample_rate: int) -> AbstractAsyncContextManager[Any]:
        return ws_client.connect(
            _WS_URL,
            subprotocols=[_SUBPROTOCOL],
            additional_headers={"Authorization": f"Bearer {self._auth_token()}"},
        )

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(
            json.dumps(
                {"type": "session.update", "language": "en-US", "audio": {"format": "pcm16"}}
            )
        )

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        # Scribe has no force-finalize: session.close drops the pending turn, so
        # feed trailing silence until the endpointer fires (cap; breaks early).
        tx.final_event.clear()
        silence = bytes(int(tx.byte_rate * _EOT_SILENCE_S))
        async for chunk, _ in paced_chunks(silence, tx.chunk_size, tx.byte_rate):
            if tx.final_event.is_set():
                break
            await ws.send(chunk)
        await ws.send(json.dumps({"type": "session.close"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type = msg.get("type")

        if msg_type == "error":
            error: dict[str, Any] = msg.get("error") or {}
            if not error.get("fatal"):
                self._log.warning("zoom_stt_warning", msg=msg)
                return False
            code = error.get("code")
            tx.fail(str(error.get("message") or f"Zoom Scribe error (code {code})"))
            self._log.warning("zoom_stt_error", msg=msg)
            return True

        if msg_type == "transcription.completed":
            # Deltas are ignored: an interim must never count as a first token.
            text = str(msg.get("transcript", "")).strip()
            if text:
                tx.final(text, now)
            else:
                tx.final_event.set()
            return False

        return bool(msg_type == "session.closed")
