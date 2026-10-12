# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Murf TTS provider — WebSocket streaming via the stream-input API."""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from urllib.parse import urlencode

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

_VALID_MODELS = ("falcon-2",)
_WIRE_MODELS = {"falcon-2": "FALCON"}
_WS_URL = "wss://us-east.api.murf.ai/v1/speech/stream-input"
_SAMPLE_RATE = 24000


class MurfTTSProvider(TTSProvider):
    """Murf TTS provider using WebSocket streaming (JSON frames, base64 audio)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Murf TTS model {model!r}. Valid: {_VALID_MODELS}")
        if not voice:
            raise ValueError("Murf TTS requires a voice")
        self._model = model
        self._voice = voice

        api_key_secret = settings.murfai_api_key
        if api_key_secret is None:
            raise ValueError("murfai_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"murf-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        synthesis = Synthesis("murf", self._model, self._voice, _SAMPLE_RATE)

        query = urlencode(
            {
                "api-key": self._api_key,
                "model": _WIRE_MODELS[self._model],
                "sample_rate": _SAMPLE_RATE,
                "channel_type": "MONO",
                "format": "PCM",
            }
        )

        try:
            async with ws_client.connect(f"{_WS_URL}?{query}") as ws:
                await ws.send(
                    json.dumps({"voice_config": {"voiceId": self._voice, "locale": "en-US"}})
                )

                synthesis.start = time.monotonic()
                await ws.send(json.dumps({"text": text, "end": True}))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)

                    error = event.get("error") or event.get("errorMessage")
                    if error:
                        raise RuntimeError(str(error))

                    synthesis.add_chunk(base64.b64decode(event.get("audio", "")))

                    if event.get("final"):
                        break

        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
