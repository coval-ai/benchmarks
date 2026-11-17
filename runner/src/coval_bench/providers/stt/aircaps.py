# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""AirCaps Live real-time STT provider.

Each clip first asks ``api.aircaps.com`` for the WebSocket URL, then opens it
with the same Bearer key. A ``start`` frame is answered by ``ready``; audio is
raw 16 kHz PCM. Turns end on our ``finalize``, and ``finish`` closes the session
with one last final and ``finished``. Every ``transcript`` frame carries the
whole session's text, so the latest final replaces the earlier ones.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_ENDPOINT_URL = "https://api.aircaps.com/v1/realtime/endpoint"
_LOOKUP_TIMEOUT_S = 10.0
_READY_TIMEOUT_S = 15.0


def _error_message(msg: dict[str, Any]) -> str:
    detail = str(msg.get("message") or "AirCaps STT error")
    code = msg.get("code")
    return f"{code}: {detail}" if code else detail


class AircapsSTTProvider(WebSocketSTTProvider):
    """AirCaps Live streaming STT provider."""

    _SAMPLE_RATE = 16000

    def __init__(self, settings: Settings, model: str = "aircaps-live-1") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("aircaps_api_key")

    @property
    def name(self) -> str:
        return "aircaps"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    async def _ws_endpoint(self) -> str:
        async with httpx.AsyncClient(timeout=_LOOKUP_TIMEOUT_S) as client:
            resp = await client.get(
                _ENDPOINT_URL, params={"model": self._model}, headers=self._headers()
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"AirCaps endpoint lookup failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        url = resp.json().get("websocket_url")
        if not url:
            raise RuntimeError("AirCaps endpoint lookup returned no websocket_url")
        return str(url)

    async def _connect(self, sample_rate: int) -> AbstractAsyncContextManager[Any]:
        url = await self._ws_endpoint()
        return ws_client.connect(url, additional_headers=self._headers())

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "model": self._model,
                    "encoding": "pcm_s16le",
                    "sample_rate": sample_rate,
                    "channels": 1,
                }
            )
        )
        try:
            async with asyncio.timeout(_READY_TIMEOUT_S):
                while True:
                    try:
                        raw = await ws.recv()
                    except StopAsyncIteration:
                        raise RuntimeError("AirCaps stream closed before ready") from None
                    if isinstance(raw, bytes):
                        continue
                    msg: dict[str, Any] = json.loads(raw)
                    msg_type = str(msg.get("type", ""))
                    if msg_type == "ready":
                        return
                    if msg_type == "error":
                        raise RuntimeError(f"AirCaps session refused: {_error_message(msg)}")
        except TimeoutError as exc:
            raise RuntimeError(
                f"AirCaps did not signal ready within {_READY_TIMEOUT_S:.0f}s"
            ) from exc

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await self._send_and_wait(ws, json.dumps({"type": "finalize"}), tx)
        await ws.send(json.dumps({"type": "finish"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type = str(msg.get("type", ""))

        if msg_type == "transcript":
            text = str(msg.get("text") or "").strip()
            if msg.get("is_final"):
                tx.finals.clear()
                tx.final(text, now)
            elif text:
                tx.partial(text, now)
        elif msg_type == "finished":
            return True
        elif msg_type == "error":
            self._log.warning("aircaps_stt_error", msg=msg)
            tx.fail(_error_message(msg))
            return True
        return False
