# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradium TTS provider — WebSocket streaming.

Wire protocol: WebSocket, wss://api.gradium.ai/api/speech/tts
Auth: x-api-key: <key>
Setup: {"type":"setup","voice_id":"...","model_name":"default","output_format":"pcm"}
Text:  {"type":"text","text":"..."}
Close: {"type":"end_of_stream"}
Audio: server sends {"type":"audio","audio":"<base64 PCM>"} chunks
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_WS_URL = "wss://api.gradium.ai/api/speech/tts"
_SAMPLE_RATE = 48000


class GradiumTTSProvider(TTSProvider):
    """Gradium TTS provider using WebSocket streaming."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

        api_key_secret = settings.gradium_tts_api_key
        if api_key_secret is None:
            raise ValueError("gradium_tts_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return "gradium"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        synthesis = Synthesis("gradium", self._model, self._voice, _SAMPLE_RATE)

        try:
            headers = {"x-api-key": self._api_key}

            async with ws_client.connect(_WS_URL, additional_headers=headers) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "setup",
                            "voice_id": self._voice,
                            "model_name": self._model,
                            "output_format": "pcm",
                        }
                    )
                )

                # Wait for ready
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg: dict[str, Any] = json.loads(raw)
                if msg.get("type") != "ready":
                    logger.warning("gradium_unexpected_first_message", msg=msg)

                synthesis.start = time.monotonic()
                await ws.send(json.dumps({"type": "text", "text": text}))
                await ws.send(json.dumps({"type": "end_of_stream"}))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    msg_type: str = msg.get("type", "")

                    if msg_type == "audio":
                        synthesis.add_chunk(base64.b64decode(str(msg.get("audio", ""))))

                    elif msg_type == "end_of_stream":
                        break

                    elif msg_type == "error":
                        raise RuntimeError(str(msg.get("message", msg)))

        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
