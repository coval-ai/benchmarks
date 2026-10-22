# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Soniox real-time STT provider (WebSocket)."""

from __future__ import annotations

import json
from typing import Any

from coval_bench.config import Settings
from coval_bench.providers.stt._streaming import Transcription, WebSocketSTTProvider

_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


class SonioxSTTProvider(WebSocketSTTProvider):
    """Soniox streaming STT provider."""

    # Tokens carry their own leading spaces.
    _FINAL_JOIN = ""
    _SAMPLE_RATE = 16000

    def __init__(self, settings: Settings, model: str = "stt-rt-v5") -> None:
        super().__init__(settings, model)
        self._api_key = self._require_secret("soniox_api_key")

    @property
    def name(self) -> str:
        return "soniox"

    def _ws_url(self, sample_rate: int) -> str:
        return _WS_URL

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    async def _open(self, ws: Any, tx: Transcription, sample_rate: int) -> None:
        await ws.send(
            json.dumps(
                {
                    "model": self._model,
                    "audio_format": "pcm_s16le",
                    "sample_rate": sample_rate,
                    "num_channels": 1,
                    "language_hints": ["en"],
                }
            )
        )

    async def _end_audio(self, ws: Any, tx: Transcription) -> None:
        # End-of-audio is an empty *text* frame; a zero-length binary frame is
        # ignored, so the server never finalizes (stalls to its idle timeout).
        await ws.send("")

    def _on_message(self, msg: Any, now: float, tx: Transcription) -> bool:
        if msg.get("error_code") or msg.get("error_message"):
            code = msg.get("error_code")
            tx.fail(str(msg.get("error_message") or f"Soniox STT error (code {code})"))
            self._log.warning("soniox_stt_error", msg=msg)
            return True

        for token in msg.get("tokens") or []:
            text = str(token.get("text", ""))
            if not text:
                continue
            # Standalone " " tokens carry no content for TTFT or partials but
            # must reach the transcript.
            if token.get("is_final"):
                tx.final(text, now)
            else:
                tx.partial(text, now)

        return bool(msg.get("finished"))
