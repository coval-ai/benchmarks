# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared base for Realtime-API-shaped STT sockets (OpenAI, Nari, StepFun).

The wire shape: Bearer auth, a session config frame answered by a ready event,
base64 PCM16 ``input_audio_buffer.append`` frames, and a manual
``input_audio_buffer.commit`` that finalizes the buffer.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, ClassVar

from pydantic import SecretStr

from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider


class RealtimeSTTProvider(WebSocketSTTProvider):
    """Manual-commit Realtime transcription session."""

    _LABEL: ClassVar[str] = ""
    _PARTIAL_FALLBACK = "longest"
    _NO_FINAL_ERROR = "ws_closed_without_completed"
    _api_key: SecretStr

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    def _frame(self, event_type: str, **fields: Any) -> str:
        return json.dumps({"type": event_type, **fields})

    def _error_message(self, event: dict[str, Any]) -> str:
        raise NotImplementedError

    async def _await_ready(self, ws: Any, ready_type: str, timeout: float) -> None:
        """Drain the handshake until *ready_type* arrives; raise on error or timeout."""
        not_ready = f"Did not receive {ready_type} within {timeout}s"
        try:
            async with asyncio.timeout(timeout):
                while True:
                    try:
                        raw = await ws.recv()
                    except StopAsyncIteration:
                        raise RuntimeError(not_ready) from None
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)
                    event_type = str(event.get("type", ""))
                    if event_type == ready_type:
                        return
                    if event_type == "error":
                        detail = self._error_message(event)
                        raise RuntimeError(f"{self._LABEL} error during session setup: {detail}")
        except TimeoutError as exc:
            raise RuntimeError(not_ready) from exc

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        await ws.send(
            self._frame("input_audio_buffer.append", audio=base64.b64encode(chunk).decode("utf-8"))
        )

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        await ws.send(self._frame("input_audio_buffer.commit"))
