# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Soniox real-time TTS streaming provider."""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from uuid import uuid4

import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import Synthesis

_VALID_VOICES = (
    "Maya",
    "Daniel",
    "Noah",
    "Nina",
    "Emma",
    "Jack",
    "Adrian",
    "Claire",
    "Grace",
    "Owen",
    "Mina",
    "Kenji",
    "Rafael",
    "Mateo",
    "Lucia",
    "Sofia",
    "Oliver",
    "Arthur",
    "Isla",
    "Victoria",
    "Cooper",
    "Mason",
    "Ruby",
    "Elise",
    "Arjun",
    "Rohan",
    "Priya",
    "Meera",
)
_WS_URL = "wss://tts-rt.soniox.com/tts-websocket"
_SAMPLE_RATE = 24000


class SonioxTTSProvider(TTSProvider):
    """Soniox TTS provider using WebSocket streaming (JSON frames, base64 audio)."""

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if voice not in _VALID_VOICES:
            raise ValueError(f"Invalid Soniox TTS voice {voice!r}. Valid: {_VALID_VOICES}")
        self._model = model
        self._voice = voice

        api_key_secret = settings.soniox_api_key
        if api_key_secret is None:
            raise ValueError("soniox_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"soniox-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        synthesis = Synthesis("soniox", self._model, self._voice, _SAMPLE_RATE)
        stream_id = str(uuid4())

        try:
            async with ws_client.connect(_WS_URL) as ws:
                synthesis.start = time.monotonic()
                # Soniox authenticates in-band: the api_key rides the opening config
                # frame rather than an Authorization header.
                await ws.send(
                    json.dumps(
                        {
                            "api_key": self._api_key,
                            "model": self._model,
                            "language": "en",
                            "voice": self._voice,
                            "audio_format": "pcm_s16le",
                            "sample_rate": _SAMPLE_RATE,
                            "stream_id": stream_id,
                        }
                    )
                )
                await ws.send(json.dumps({"text": text, "text_end": True, "stream_id": stream_id}))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    event: dict[str, Any] = json.loads(raw)

                    if event.get("error_code") or event.get("error_message"):
                        message = event.get("error_message") or (
                            f"Soniox TTS error (code {event.get('error_code')})"
                        )
                        raise RuntimeError(str(message))

                    synthesis.add_chunk(base64.b64decode(event.get("audio") or ""))

                    if event.get("terminated"):
                        break

        except Exception as exc:
            synthesis.fail(exc)

        return synthesis.result()
