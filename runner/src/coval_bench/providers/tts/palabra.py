# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Palabra real-time TTS streaming provider."""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

_VALID_VOICES = ("default_low", "default_high")
_WS_URL = "wss://stream.us.palabra.ai/tts-api/v1/text-to-speech/stream"
_SAMPLE_RATE = 24000


class PalabraTTSProvider(TTSProvider):
    """Palabra TTS provider using WebSocket streaming (JSON frames, base64 audio)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if voice not in _VALID_VOICES:
            raise ValueError(f"Invalid Palabra TTS voice {voice!r}. Valid: {_VALID_VOICES}")
        self._model = model
        self._voice = voice

        api_key_secret = settings.palabra_api_key
        if api_key_secret is None:
            raise ValueError("palabra_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return "palabra"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        synthesis = Synthesis("palabra", self._model, self._voice, _SAMPLE_RATE)

        try:
            # Platform auth: the API key authenticates the WebSocket directly.
            token = quote(self._api_key, safe="")
            async with ws_client.connect(f"{_WS_URL}?token={token}") as ws:
                # Pre-t0: session init
                await ws.send(
                    json.dumps(
                        {
                            "type": "init",
                            "language": "en",
                            "model": self._model,
                            "voice_options": {"voice_id": self._voice},
                            "output": {"format": "pcm", "sample_rate": _SAMPLE_RATE},
                        }
                    )
                )

                synthesis.start = time.monotonic()
                generation_id = f"coval_{uuid4().hex}"
                await ws.send(
                    json.dumps(
                        {
                            "type": "text",
                            "text": text,
                            "generation_id": generation_id,
                            "is_eos": True,
                        }
                    )
                )

                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)

                    data_raw = event.get("data")
                    data: dict[str, Any] = data_raw if isinstance(data_raw, dict) else {}

                    if event.get("message_type") == "error":
                        raise RuntimeError(f"{data.get('code')}: {data.get('desc')}")

                    if event.get("message_type") != "audio_chunk":
                        continue

                    synthesis.add_chunk(base64.b64decode(data.get("audio", "")))

                    if data.get("last_chunk"):
                        break

        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
