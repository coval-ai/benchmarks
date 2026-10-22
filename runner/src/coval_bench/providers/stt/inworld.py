# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Inworld AI real-time STT provider (WebSocket bidirectional streaming).

INWORLD_API_KEY is the portal-issued base64 credential; it rides the
``Authorization: Basic <key>`` header verbatim (re-encoding it fails auth).
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any, cast

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_URL = "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional"

# Inworld rejects chunks outside 20-1000 ms with "invalid audio chunk duration".
_MIN_CHUNK_MS = 20
_MIN_CHUNK_BYTES = 16000 * 2 * _MIN_CHUNK_MS // 1000


@dataclass
class _InworldTranscription(Transcription):
    # Set on the trailing usage frame; the sender closes on it.
    done: asyncio.Event = field(default_factory=asyncio.Event)


class InworldSTTProvider(WebSocketSTTProvider):
    """Inworld AI streaming STT provider."""

    _SAMPLE_RATE = 16000
    _MIN_TAIL_BYTES = _MIN_CHUNK_BYTES
    _NO_FINAL_ERROR = "Inworld stream ended before a final transcription was received"

    def __init__(self, settings: Settings, model: str = "inworld-stt-1") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("inworld_api_key")

    @property
    def name(self) -> str:
        return "inworld"

    def _new_transcription(self, result: TranscriptionResult) -> Transcription:
        return _InworldTranscription(result)

    def _ws_url(self, sample_rate: int) -> str:
        return _WS_URL

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Basic {self._api_key.get_secret_value()}"}

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        # Inworld routes by a "{provider}/{model}" id; we own the inworld/ namespace.
        await ws.send(
            json.dumps(
                {
                    "transcribeConfig": {
                        "modelId": f"inworld/{self._model}",
                        "audioEncoding": "LINEAR16",
                        "sampleRateHertz": sample_rate,
                        "numberOfChannels": 1,
                        "language": "en-US",
                    }
                }
            )
        )

    def _chunk_size(self, byte_rate: int, realtime_resolution: float) -> int:
        # Inworld accepts 20-1000 ms chunks.
        return min(max(int(byte_rate * realtime_resolution), _MIN_CHUNK_BYTES), byte_rate)

    async def _send_chunk(self, ws: Any, chunk: bytes) -> None:
        await ws.send(json.dumps({"audioChunk": {"content": base64.b64encode(chunk).decode()}}))

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        tx = cast(_InworldTranscription, tx)
        # No endTurn: server-side turn detection is on, and endTurn + closeStream
        # makes Inworld append a phantom final segment.
        await self._send_and_wait(ws, json.dumps({"closeStream": {}}), tx, event=tx.done)
        await ws.close()

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        tx = cast(_InworldTranscription, tx)
        if msg.get("error"):
            tx.fail(str(msg["error"]))
            self._log.warning("inworld_stt_error", msg=msg)
            return True
        if "usage" in msg:
            tx.done.set()
            return True

        transcription = msg.get("result", {}).get("transcription")
        if not transcription:
            return False
        text = str(transcription.get("transcript", "")).strip()
        if not text:
            return False

        # Inworld emits whole-phrase final segments; the template joins them with spaces.
        if transcription.get("isFinal"):
            tx.final(text, now)
        else:
            tx.partial(text, now)
        return False
