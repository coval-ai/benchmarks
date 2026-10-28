# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Krisp real-time STT provider (WebSocket, beta).

Auth is two-step: the API key mints a short-lived session token over HTTP, and
only that token goes on the socket. Each transcript frame carries word pieces
with their own leading spaces; finals are committed once, partials cover only
the region after the last final.
Protocol: https://sdk-docs.krisp.ai/docs/realtime-stt
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

_TOKEN_URL = "https://api.developers.krisp.ai/v2/sdk/tokens/session"  # noqa: S105
_WS_URL = "wss://streaming.krisp.ai/stt"
_TOKEN_TTL_MIN = 5  # Krisp's floor
_TOKEN_TIMEOUT_S = 10.0
_READY_TIMEOUT_S = 15.0
# The server wants 80 ms chunks (or multiples), each under 64 KiB.
_CHUNK_BYTES = 16000 * 2 * 80 // 1000
_MAX_MESSAGE_BYTES = 65536


def _error_message(msg: dict[str, Any]) -> str:
    error = msg.get("error") or {}
    detail = str(error.get("message") or error.get("code") or "Krisp STT error")
    request_id = msg.get("request_id")
    return f"{detail} (request_id={request_id})" if request_id else detail


class KrispSTTProvider(WebSocketSTTProvider):
    """Krisp streaming STT provider for the agents and humans models."""

    _FINAL_JOIN = ""
    _SAMPLE_RATE = 16000

    def __init__(self, settings: Settings, model: str = "krisp-realtime-stt-agents-v1") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("krisp_api_key")

    @property
    def name(self) -> str:
        return "krisp"

    async def _session_token(self) -> str:
        headers = {"Authorization": f"api-key {self._api_key.get_secret_value()}"}
        body = {"usage_type": "stt", "expiration_ttl": _TOKEN_TTL_MIN}
        async with httpx.AsyncClient(timeout=_TOKEN_TIMEOUT_S) as client:
            resp = await client.post(_TOKEN_URL, headers=headers, json=body)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Krisp session token failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        token = (resp.json().get("data") or {}).get("session_key")
        if not token:
            raise RuntimeError("Krisp session token response had no session_key")
        return str(token)

    async def _connect(self, sample_rate: int) -> AbstractAsyncContextManager[Any]:
        token = await self._session_token()
        return ws_client.connect(_WS_URL, additional_headers={"Authorization": f"api-key {token}"})

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(
            json.dumps(
                {
                    "model": self._model,
                    "sample_rate": sample_rate,
                    "channels": 1,
                    "audio_format": "pcm_s16le",
                    "language_hints": ["en"],
                }
            )
        )
        try:
            async with asyncio.timeout(_READY_TIMEOUT_S):
                while True:
                    try:
                        raw = await ws.recv()
                    except StopAsyncIteration:
                        raise RuntimeError("Krisp stream closed before ready") from None
                    if isinstance(raw, bytes):
                        continue
                    msg: dict[str, Any] = json.loads(raw)
                    msg_type = str(msg.get("type", ""))
                    if msg_type == "ready":
                        return
                    if msg_type == "error":
                        raise RuntimeError(f"Krisp session setup failed: {_error_message(msg)}")
        except TimeoutError as exc:
            raise RuntimeError(
                f"Krisp did not signal ready within {_READY_TIMEOUT_S:.0f}s"
            ) from exc

    def _chunk_size(self, byte_rate: int, realtime_resolution: float) -> int:
        frames = max(1, int(byte_rate * realtime_resolution) // _CHUNK_BYTES)
        return min(frames * _CHUNK_BYTES, _MAX_MESSAGE_BYTES // _CHUNK_BYTES * _CHUNK_BYTES)

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        # The server flushes the remaining finals, sends end, and closes itself.
        await ws.send(json.dumps({"type": "end_of_audio"}))

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        msg_type = str(msg.get("type", ""))

        if msg_type == "error":
            tx.fail(_error_message(msg))
            self._log.warning("krisp_stt_error", msg=msg)
            return True

        if msg_type == "end":
            return True

        if msg_type != "transcript":
            return False

        transcript: dict[str, Any] = msg.get("transcript") or {}
        text = "".join(str(word.get("text", "")) for word in transcript.get("words") or [])
        if not text:
            return False
        if transcript.get("isFinal"):
            tx.final(text, now)
        else:
            tx.partial(text, now)
        return False
