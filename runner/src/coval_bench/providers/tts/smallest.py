# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Smallest AI Lightning TTS provider — WebSocket streaming.

Wire protocol:
  connect → send JSON(text, voice_id, model, sample_rate, language)
  → recv JSON chunks: {"status": "chunk", "data": {"audio": "<base64-pcm>"}}
  → recv JSON done:   {"status": "complete", "done": true}

Auth:   Authorization: Bearer <key>
Output: 16-bit mono PCM (base64-decoded from JSON envelopes), 24 kHz
"""

from __future__ import annotations

import base64
import json
import time

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

SAMPLE_RATE = 24000
_WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"


class SmallestTTSProvider(TTSProvider):
    """Smallest AI Lightning TTS provider using WebSocket streaming."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

        api_key_secret = settings.smallest_api_key
        if api_key_secret is None:
            raise ValueError("smallest_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"smallest-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via Smallest AI WebSocket and return a TTSResult."""
        synthesis = Synthesis("smallest", self._model, self._voice, SAMPLE_RATE)

        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = json.dumps(
            {
                "text": text,
                "voice_id": self._voice,
                "model": self._model,
                "sample_rate": SAMPLE_RATE,
                # Explicit language prevents auto-detect flakiness on English-only models.
                "language": "en",
            }
        )

        try:
            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
                synthesis.start = time.monotonic()
                await ws.send(payload)

                async for raw in ws:
                    if isinstance(raw, bytes):
                        # Graceful fallback: accept raw binary if server ever sends it.
                        synthesis.add_chunk(raw)
                        continue

                    msg = json.loads(raw)
                    status = msg.get("status", "")

                    if status == "chunk":
                        synthesis.add_chunk(base64.b64decode(msg.get("data", {}).get("audio", "")))
                    elif status == "complete" or msg.get("done"):
                        break

        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
